// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Parser.h
///
/// @brief Defines various classes (Processor, Parser, Option, Action, Loop)
///        for processing of command-line arguments.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_PARSER_HAS_BEEN_INCLUDED
#define VDB_TOOL_PARSER_HAS_BEEN_INCLUDED

#include <iostream>
#include <sstream>
#include <string> // for std::string, std::stof and std::stoi
#include <algorithm> // std::sort
#include <random>
#include <functional>
#include <vector>
#include <list>
#include <set>
#include <time.h>
#include <initializer_list>
#include <unordered_map>
#include <iterator>// for std::advance
#include <sys/stat.h>
#include <stdio.h>

#include <openvdb/openvdb.h>
#include <openvdb/util/Assert.h>

#include "Util.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

// ==============================================================================================================

/// @brief This class defines string attributes for options, i.e. arguments for actions
struct Option {
    void append(const std::string &v) {value = value.empty() ? v : value + "," + v;}

    std::string name, value, example, documentation;
};

// ==============================================================================================================
struct Action {
    /// @brief c-tor
    Action(std::string _name,
           std::string _alias,
           std::string _doc,
           std::vector<Option> &&_options,
           std::function<void()> &&_init,
           std::function<void()> &&_run,
           size_t _anonymous = -1)
      : name(std::move(_name))
      , alias(std::move(_alias))
      , documentation(std::move(_doc))
      , anonymous(_anonymous)
      , options(std::move(_options))
      , init(std::move(_init))
      , run(std::move(_run)) {}
    /// @brief default copy constructor
    Action(const Action&) = default;
    /// @brief Sets the options of this actions
    void setOption(const std::string &str);
    void print(std::ostream& os = std::cerr) const;

    std::string            name;// primary name of action, eg "read"
    std::string            alias;// alternate name for action, eg "i"
    std::string            documentation;// documentation e.g. "read", "i", "files", "read files"
    size_t                 anonymous;// index of the option to which the value of un-named option will be appended, e.g. files
    std::vector<Option>    options;// e.g. {{"grids", "density,sphere"}, {"files", "path/base.ext"}}
    std::function<void()>  init, run;// callback functions
};// Action struct

// ==============================================================================================================

using ActListT = std::list<Action>;
using ActIterT = typename ActListT::iterator;
using VecF = std::vector<float>;// vector of floats
using VecI = std::vector<int>;// vector for integers
using VecS = std::vector<std::string>;// vector of strings

// ==============================================================================================================

/// @brief Class that stores values by name
class Memory
{
public:

    Memory() = default;
    std::string get(const std::string &name) {
        auto it = mData.find(name);
        if (it == mData.end()) {
            if (name=="pi") {
                return std::to_string(std::atan(1)*4);
            } else if (name=="e") {
                return std::to_string(2.718281828459);
            } else {
                throw std::invalid_argument("Storrage::get: undefined variable \""+name+"\"");
            }
        }
        return it->second;
    }
    void clear() {mData.clear();}
    void clear(const std::string &name) {mData.erase(name);}
    void set(const std::string &name, const std::string &value) {mData[name]=value;}
    void set(const std::string &name, const char *value) {mData[name]=value;}
    template <typename T>
    void set(const std::string &name, const T &value) {mData[name]=std::to_string(value);}
    void print(std::ostream& os = std::cerr) const {
        std::map<std::string, std::string> tmp(mData.begin(),mData.end());// sort output
        for (auto &d : tmp) os << d.first <<"="<<d.second<<std::endl;
    }
    size_t size() const {return mData.size();}
    bool isSet(const std::string &name) const {return mData.find(name)!=mData.end();}

private:

    std::unordered_map<std::string, std::string> mData;
};// Memory

// ==============================================================================================================

class Stack {
public:

    Stack(){mData.reserve(10);}
    Stack(std::initializer_list<std::string> d) : mData(d.begin(), d.end()) {}
    size_t depth() const {return mData.size();}
    bool empty() const {return mData.empty();}
    bool operator==(const Stack &other) const {return mData == other.mData;}
    void push(const std::string &s) {mData.push_back(s);}
    std::string pop() {// y x -- y
        if (mData.empty()) throw std::invalid_argument("Stack::pop: empty stack");
        const std::string str = mData.back();
        mData.pop_back();
        return str;
    }
    void drop() {// y x -- y
        if (mData.empty()) throw std::invalid_argument("Stack::drop: empty stack");
        mData.pop_back();
    }
    std::string& top() {
        if (mData.empty()) throw std::invalid_argument("Stack::top: empty stack");
        return mData.back();
    }
    const std::string& peek() const {
        if (mData.empty()) throw std::invalid_argument("Stack::peak: empty stack");
        return mData.back();
    }
    void dup() {// x -- x x
        if (mData.empty()) throw std::invalid_argument("Stack::dup: empty stack");
        mData.push_back(mData.back());
    }
    void swap() {// y x -- x y
        if (mData.size()<2) throw std::invalid_argument("Stack::swap: size<2");
        const size_t n = mData.size()-1;
        std::swap(mData[n], mData[n-1]);
    }
    void nip() {// y x -- x
        if (mData.size()<2) throw std::invalid_argument("Stack::nip: size<2");
        mData.erase(mData.end()-2);
    }
    void scrape() {// ... x -- x
        if (mData.empty()) throw std::invalid_argument("Stack::scrape: empty stack");
        mData.erase(mData.begin(), mData.end()-1);
    }
    void clear() {mData.clear();}
    void over() {// y x -- y x y
        if (mData.size()<2) throw std::invalid_argument("Stack::over: size<2");
        mData.push_back(mData[mData.size()-2]);
    }
    void rot() {// z y x -- y x z
        if (mData.size()<3) throw std::invalid_argument("Stack::rot: size<3");
        const size_t n = mData.size() - 1;
        std::swap(mData[n-2], mData[n  ]);
        std::swap(mData[n-2], mData[n-1]);
    }
    void tuck() {// z y x -- x z y
        if (mData.size()<3) throw std::invalid_argument("Stack::tuck: size<3");
        const size_t n = mData.size()-1;
        std::swap(mData[n-2], mData[n]);
        std::swap(mData[n-1], mData[n]);
    }
    void print(std::ostream& os = std::cerr) const {
        if (mData.empty()) return;
        os << mData[0];
        for (size_t i=1; i<mData.size(); ++i) os << "," << mData[i];
    }

private:

    std::vector<std::string> mData;
};// Stack

// ==============================================================================================================

/// @brief   Implements a light-weight stack-oriented programming language (very loosely) inspired by Forth
/// @details Specifically, it uses Reverse Polish Notation to define operations that are evaluated during
///          paring of the command-line arguments (options to be precise).
class Processor
{
public:

    template <typename T>
    void push(const T &t) {mCallStack.push(std::to_string(t));}
    void push(const std::string &s) {mCallStack.push(s);}
    template <typename T>
    void set(const T &t) {mCallStack.top() = std::to_string(t);}
    void set(bool t) {mCallStack.top() = t ? "1" : "0";}
    void set(const std::string &str) {mCallStack.top() = str;}
    void set(const char *str) {mCallStack.top() = str;}
    std::string& get() {return mCallStack.top();}
    const Memory& memory() const {return mMemory;}
    Memory& memory() {return mMemory;}
    void add(const std::string &name, std::string &&doc, std::function<void()> &&func) {mInstructions[name]={std::move(doc),std::move(func)};}

    /// @brief default c-tor for Processor. Add all the instructions required by the tool
    Processor()
    {
        // file-name operations
        add("path","extract file path from string, e.g. {path/base0123.ext:path} -> {path}",
            [&](){mCallStack.top()=getPath(mCallStack.top());});
        add("file","extract file name from string, e.g. {path/base0123.ext:file} -> {base0123.ext}",
            [&](){mCallStack.top()=getFile(mCallStack.top());});
        add("name","extract file name without extension from string, e.g. {path/base0123:name} -> {extbase0123}",
            [&](){mCallStack.top()=getName(mCallStack.top());});
        add("base","extract file base name from string, e.g. {path/base0123.ext:base -> {base}",
            [&](){mCallStack.top()=getBase(mCallStack.top());});
        add("number","extract file number from string, e.g. {path/base0123.ext:number} -> {0123}",
            [&](){mCallStack.top()=getNumber(mCallStack.top());});
        add("ext","extract file extension from string, e.g. {path/base0123.ext:ext} -> {ext}",
            [&](){mCallStack.top()=getExt(mCallStack.top());});

        // boolean operations
        add("==","returns true if the two top enteries on the stack compare equal, e.g. {1:2:==} -> {0}",
            [&](){this->boolean(std::equal_to<>());});
        add("!=","returns true if the two top enteries on the stack are not equal, e.g. {1:2:!=} -> {1}",
            [&](){this->boolean(std::not_equal_to<>());});
        add("<=","returns true if the two top enteries on the stack are less than or equal, e.g. {1:2:<=} -> {1}",
            [&](){this->boolean(std::less_equal<>());});
        add(">=","returns true if the two top enteries on the stack are greater than or equal, e.g. {1:2:>=} -> {0}",
            [&](){this->boolean(std::greater_equal<>());});
        add("<","returns true if the two top enteries on the stack are less than, e.g. {1:2:<} -> {1}",
            [&](){this->boolean(std::less<>());});
        add(">","returns true if the two top enteries on the stack are less than or equal, e.g. {1:2:<=} -> {1}",
            [&](){this->boolean(std::greater<>());});
        add("!","logical negation, e.g. {1:!} -> {0}",
            [&](){this->set(!strToBool(mCallStack.top()));});
        add("|","logical or, e.g. {1:0:|} -> {1}",
            [&](){bool b=strToBool(mCallStack.pop());this->set(strToBool(mCallStack.top())||b);});
        add("&","logical and, e.g. {1:0:&} -> {0}",
            [&](){bool b=strToBool(mCallStack.pop());this->set(strToBool(mCallStack.top())&&b);});

        // math operations
        add("+","adds two top stack entries, e.g. {1:2:+} -> {3}",
            [&](){this->ab(std::plus<>());});
        add("-","subtracts two top stack entries, e.g. {1:2:-} -> {-1}",
            [&](){this->ab(std::minus<>());});
        add("*","multiplies two top stack entries, e.g. {1:2:*} -> {2}",
            [&](){this->ab(std::multiplies<>());});
        add("/","adds two top stack entries, e.g. {1.0:2.0:/} -> {0.5} and {1:2:/} -> {0}",
            [&](){this->ab(std::divides<>());});
        add("++","increment top stack entry, e.g. {1:++} -> {2}",
            [&](){this->a([](auto& x){return ++x;});});
        add("--","decrement top stack entry, e.g. {1:--} -> {0}",
            [&](){this->a([](auto& x){return --x;});});
        add("abs","absolute value, {-1:abs} -> {1}",
            [&](){this->a([](auto& x){return math::Abs(x);});});
        add("ceil","ceiling of floating point value, e.g. {0.5:ceil} -> {0.0}",
            [&](){this->a([](auto& x){return std::ceil(x);});});
        add("floor","floor of floating point value, e.g. {0.5:floor} -> {1.0}",
            [&](){this->a([](auto& x){return std::floor(x);});});
        add("pow2","square of value, e.g. {2:pow2} -> {4}",
            [&](){this->a([](auto& x){return math::Pow2(x);});});
        add("pow3","cube of value, e.g. {2:pow3} -> {8}",
            [&](){this->a([](auto& x){return math::Pow3(x);});});
        add("pow","power of vale, e.g. {2:3:pow} -> {8}",
            [&](){this->ab([](auto& a, auto& b){return math::Pow(a, b);});});
        add("min","minimum of two values, e.g. {1:2:min} -> {1}",
            [&](){this->ab([](auto& a, auto& b){return std::min(a, b);});});
        add("max","minimum of two values, e.g. {1:2:max} -> {2}",
            [&](){this->ab([](auto& a, auto& b){return std::max(a, b);});});
        add("neg","negative of value, e.g. {1:neg} -> {-1}",
            [&](){this->a([](auto& x){return -x;});});
        add("sign","sign of value, e.g. {-2:neg} -> {-1}",
            [&](){this->a([](auto& x){return (x > 0) - (x < 0);});});
        add("sin","sine of value, e.g. {$pi:sin} -> {0.0}",
            [&](){this->set(std::sin(strToFloat(mCallStack.top())));});
        add("cos","cosine of value, e.g. {$pi:cos} -> {-1.0}",
            [&](){this->set(std::cos(strToFloat(mCallStack.top())));});
        add("tan","tangent of value, e.g. {$pi:tan} -> {0.0}",
            [&](){this->set(std::tan(strToFloat(mCallStack.top())));});
        add("asin","inverse sine of value, e.g. {1:asin} -> {1.570796}",
            [&](){this->set(std::asin(strToFloat(mCallStack.top())));});
        add("acos","inverse cosine of value, e.g. {1:acos} -> {0.0}",
            [&](){this->set(std::acos(strToFloat(mCallStack.top())));});
        add("atan","inverse tangent of value, e.g. {1:atan} -> {0.785398}",
            [&](){this->set(std::atan(strToFloat(mCallStack.top())));});
        add("r2d","radian to degrees, e.g. {$pi:r2d} -> {180.0}",
            [&](){this->set(180.0f*strToFloat(mCallStack.top())/math::pi<float>());});
        add("d2r","degrees to radian, e.g. {180:d2r} -> {3.141593}",
            [&](){this->set(math::pi<float>()*strToFloat(mCallStack.top())/180.0f);});
        add("inv","inverse of value, e.g. {5:inv} -> {0.2}",
            [&](){this->set(1.0f/strToFloat(mCallStack.top()));});
        add("exp","exponential of value, e.g. {1:exp} -> {2.718282}",
            [&](){this->set(std::exp(strToFloat(mCallStack.top())));});
        add("ln","natural log of value, e.g. {1:ln} -> {0.0}",
            [&](){this->set(std::log(strToFloat(mCallStack.top())));});
        add("log","10 base log of value, e.g. {1:log} -> {0.0}",
            [&](){this->set(std::log10(strToFloat(mCallStack.top())));});
        add("sqrt","squareroot of value, e.g. {2:sqrt} -> {1.414214}",
            [&](){this->set(std::sqrt(strToFloat(mCallStack.top())));});
        add("to_int","convert value to integer, e.g. {1.2:to_int} -> {1}",
            [&](){this->set(int(strToFloat(mCallStack.top())));});
        add("to_float","convert value to floating point, e.g. {1:to_float} -> {1.0}",
            [&](){this->set(strToFloat(mCallStack.top()));});

        // stack operations
        add("dup","duplicates the top, i.e. pushes the top entry onto the stack, e.g. {x:dup} -> {x:x}",
            [&](){mCallStack.dup();});
        add("nip","remove the entry below the top, e.g. {x:y:nip} -> {y}",
            [&](){mCallStack.nip();});
        add("drop","remove/pop the top entry, e.g. {x:y:drop} -> {x}",
            [&](){mCallStack.drop();});
        add("swap","swap the two top entries, e.g. {x:y:swap} -> {y:x}",
            [&](){mCallStack.swap();});
        add("over","push second entry onto the top, e.g. {x:y:over} -> {x:y:x}",
            [&](){mCallStack.over();});
        add("rot","rotate three top entries left, e.g. {x:y:z:rot} -> {y:z:x}",
            [&](){mCallStack.rot();});
        add("tuck","rotate three top entries right, e.g. {x:y:z:tuck} -> {z:x:y}",
            [&](){mCallStack.tuck();});
        add("scrape","removed everything except for the top, e.g. {x:y:z:scrape} -> {z}",
            [&](){mCallStack.scrape();});
        add("clear","remove everything on the stack, e.g. {x:y:z:clear} -> {}",
            [&](){mCallStack.clear();});
        add("depth","push depth of stack onto the stack, e.g. {x:y:z:depth} -> {3}",
            [&](){this->push(mCallStack.depth());});
        add("squash","combines entire stack into the top, e.g. {x:y:z:squash} -> {x,y,z}",
            [&](){if (mCallStack.empty()) return;
                  std::stringstream ss;
                  mCallStack.print(ss);
                  mCallStack.scrape();
                  mCallStack.top()=ss.str();
        });

        // string operations
        add("lower","convert all characters in a string to lower case, e.g. {HeLlO:lower} -> {hello}",
            [&](){toLowerCase(mCallStack.top());});
        add("upper","convert all characters in a string to upper case, e.g. {HeLlO:upper} -> {HELLO}",
            [&](){toUpperCase(mCallStack.top());});
        add("length","push the number of characters in a string onto the stack, e.g. {foo bar:length} -> {7}",
            [&](){this->set(mCallStack.top().length());});
        add("replace","replace words in string, e.g. {for bar:a:b:replace} -> {foo bbr}",
            [&](){std::string b = mCallStack.pop(), a = mCallStack.pop(), &t = mCallStack.top();
                  for (size_t i=a.size(),j=b.size(),p=t.find(a); p!=std::string::npos; p=t.find(a,p+j)) t.replace(p,i,b);
        });
        add("erase","remove words in string, e.g. {foo bar:a:erase} -> {foo br}",
            [&](){std::string a = mCallStack.pop(), &t = mCallStack.top();
                  for (size_t p=t.find(a), n=a.size(); p!=std::string::npos; p=t.find(a,p)) t.erase(p,n);
        });
        add("append","append string to string, e.g. {foo:bar:append} -> {foobar}",
            [&](){const std::string str = mCallStack.pop();
                  mCallStack.top() += str;
        });
        add("tokenize","split a string according to a specific delimiter and push the tokens onto the stack e.g. foo,bar:,:tokenize -> foo:bar",
            [&](){const std::string delimiters = mCallStack.pop(), str = mCallStack.pop();
                  for (auto &s : tokenize(str, delimiters.c_str())) mCallStack.push(s);
        });
        add("match","test if a word matches a string, e.g. {sphere_01.vdb:sphere:match} -> {1}",
            [&](){std::string word = mCallStack.pop();
                  this->set(mCallStack.top().find(word) != std::string::npos);
        });

        add("is_set","returns true if a string has an associated value, e.g. {pi:is_set} ->{1}",
            [&](){this->set(mMemory.isSet(mCallStack.top()));});
        add("pad0","add zero-padding of a specified with to a string, e.g. {12:4:pad0} -> {0012}",
            [&](){const int w = strToInt(mCallStack.pop());
                  std::stringstream ss;
                  ss << std::setfill('0') << std::setw(w) << mCallStack.top();
                  mCallStack.top() = ss.str();
        });

        add("get","get the value of a variable from memory, e.g. {pi:get} -> {3.141593}, equal to {$pi}",
            [&](){mCallStack.top() = mMemory.get(mCallStack.top());});
        add("set","set a variable to a value and save it to memory, e.g. {1:G:set} -> {}, equal to {1:@G}",
            [&](){const std::string str = mCallStack.pop();
                  mMemory.set(str, mCallStack.pop());
        });
        add("date","date, e.g {date} -> {Sun Mar 27 19:31:16 2022} or {date: :_:replace} -> {Sun_Mar_27_19:31:55_2022}",
            [&](){std::time_t tmp = std::time(nullptr);
                  std::stringstream ss;
                  ss << std::asctime(std::localtime(&tmp));
                  this->push(ss.str());
        });
        add("uuid","an approximate uuid v4 random hex string, e.g. {uuid} -> {821105a2-0e60-4a23-970d-0165e0ad4373}",
            [&](){this->push(uuid());}
        );

        // dummy entries for documentation
        add("$","get the value of a variable from memory, e.g. {$pi} -> {3.141593}", [](){});
        add("@","set a variable to a value and save it to memory, e.g. {1:@G} -> {}", [](){});
        add("if","if- and optional else-statement, e.g. {1:if(2)} -> {2} and {0:if(2?3)} -> {3}",[](){});
        add("switch","switch-statement, e.g. {2:switch(1:first?2:second?3:third)} -> {second}",[](){});
        add("quit","terminate evaluation, e.g. {1:2:+:quit:4:*} -> {3}",[](){});
    }

    /// @brief performs syntax analysis on the specified str. Implements control-flows like for- and each-loops, if- and switch-statements,
    ///        and calls the lambda functions associated with the instructions.
    void operator()(std::string &str)
    {
        try {
        for (size_t pos = str.find_first_of("{}"); pos != std::string::npos; pos = str.find_first_of("{}", pos)) {
            if (str[pos]=='}') throw std::invalid_argument("Processor(): expected \"{\" before \"}\" in \""+str.substr(pos)+"\"");
            size_t end = str.find_first_of("{}", pos + 1);
            if (end == std::string::npos || str[end]=='{') throw std::invalid_argument("Processor(): nested \"{}\" is not allowed in \""+str.substr(pos)+"\"");
            for (size_t p=str.find_first_of(":}",pos+1), q=pos+1; p<=end; q=p+1, p=str.find_first_of(":}",q)) {
                if (p == q) {// ignores {:} and {::}
                    continue;
                } else if (str[q]=='$') {// get value
                    mCallStack.push(mMemory.get(str.substr(q + 1, p - q - 1)));
                } else if (str[q]=='@') {// set value
                    if (mCallStack.empty()) throw std::invalid_argument("Processor::(): cannot evaluate \""+str.substr(q,p-q)+"\" when the stack is empty");
                    mMemory.set(str.substr(q + 1, p - q - 1), mCallStack.pop());
                } else if (str.compare(q,3,"if(")==0) {// if-statement: 0|1:if(a) or 0|1:if(a?b)}
                    const size_t i = str.find_first_of("(){}", q+3);
                    if (str[i]!=')') throw std::invalid_argument("Processor():: missing \")\" in if-statement \""+str.substr(q)+"\"");
                    const auto v = tokenize(str.substr(q+3, i-q-3), "?");
                    if (v.size() == 1) {
                        if (strToBool(mCallStack.pop())) {
                            str.replace(q, i - q + 1, v[0]);
                        } else {
                            str.erase(q - 1, i - q + 2);// also erase the leading ':' character
                        }
                    } else if (v.size() == 2) {
                        str.replace(q, i - q + 1, v[strToBool(mCallStack.pop()) ? 0 : 1]);
                    } else {
                        throw std::invalid_argument("Processor():: invalid if-statement \""+str.substr(q)+"\"");
                    }
                    end = str.find('}', pos + 1);// needs to be recomputed since str was modified
                    p = q - 1;// rewind
                } else if (str.compare(q,4,"quit")==0) {// quit
                    break;
                } else if (str.compare(q,7,"switch(")==0) {//switch-statement: $1:switch(a:case_a?b:case_b?c:case_c)
                    const size_t i = str.find_first_of("(){}", q+7);
                    if (str[i]!=')') throw std::invalid_argument("Processor():: missing \")\" in switch-statement \""+str.substr(q)+"\"");
                    for (auto s : tokenize(str.substr(q+7, i-q-7), "?")) {
                        const size_t j = s.find(':');
                        if (j==std::string::npos) throw std::invalid_argument("Processor():: missing \":\" in switch-statement \""+str.substr(q)+"\"");
                        if (mCallStack.top() == s.substr(0,j)) {
                            str.replace(q, i - q + 1, s.substr(j + 1));
                            end = str.find('}', pos + 1);// needs to be recomputed since str was modified
                            p = q - 1;// rewind
                            mCallStack.drop();
                            break;
                        }
                    }
                    if (str.compare(q,7,"switch(")==0) throw std::invalid_argument("Processor():: no match in switch-statement \""+str.substr(q)+"\"");
                } else {// apply callback or push
                    const std::string s = str.substr(q, p - q);
                    auto it = mInstructions.find(s);
                    if (it != mInstructions.end()) {
                        it->second.callback();
                    } else {
                        mCallStack.push(s);
                    }
                }
            }// for-loop over ":" in string
            if (mCallStack.empty()) {// if call stack is empty clear inout string
                str.erase(pos, end-pos+1);
            } else if (mCallStack.depth()==1) {// if call stack has one entry replace it with the input string
                str.replace(pos, end-pos+1, mCallStack.pop());
            } else {// more than one entry in the call stack is considered an error
                std::stringstream ss;
                mCallStack.print(ss);
                throw std::invalid_argument("Processor::(): compute stack contains more than one entry: " + ss.str());
            }
        }// for-loop over "{}" in string
        } catch (const std::exception& e) {
            throw std::invalid_argument("Error evaluating \""+str+"\": "+e.what());
        }
    }
    std::string operator()(const std::string &str)
    {
        std::string tmp = str;// copy
        (*this)(tmp);
        return tmp;
    }
    void help(std::ostream& os = std::cerr) const
    {
        std::set<std::string> vec;// print help in lexicographic order
        for (auto it=mInstructions.begin(); it!=mInstructions.end(); ++it) vec.insert(it->first);
        this->help(vec, os);
    }
    template <typename VecT>
    void help(const VecT &vec, std::ostream& os = std::cerr) const
    {
        size_t w = 0;
        for (auto &s : vec) w = std::max(w, s.size());
        w += 2;
        for (auto &s : vec) {
            auto it = mInstructions.find(s);
            if (it != mInstructions.end()) {
                os << std::left << std::setw(static_cast<int>(w)) << it->first << it->second.doc << "\n\n";
            } else {
                throw std::invalid_argument("Processor::help:: unknown operation \"" + s + "\"");
            }
        }
    }

private:

    struct Instruction {std::string doc; std::function<void()> callback;};// documentation and callback for instruction
    using Instructions = std::unordered_map<std::string, Instruction>;

    Stack        mCallStack;// processor stack for data and instructions
    Instructions mInstructions;// map of all supported instructions
    Memory       mMemory;

    /// @brief apply functor to the top element, assuming it's either a float or integer
    template <typename OpT>
    void a(OpT op){
        union {std::int32_t i; float x;} A;
        if (isInt(mCallStack.top(), A.i)) {
            mCallStack.top() = std::to_string(op(A.i));
        } else if (isFloat(mCallStack.top(), A.x)) {
            mCallStack.top() = std::to_string(op(A.x));
        } else {
            throw std::invalid_argument("a: invalid argument \"" + mCallStack.top() + "\"");
        }
    }

    /// @brief apply functor to the two top element, assuming they are both floats or integers
    template <typename OpT>
    void ab(OpT op){
        union {std::int32_t i; float x;} A, B;
        const std::string str = mCallStack.pop();
        if (isInt(mCallStack.top(), A.i) && isInt(str, B.i)) {
            mCallStack.top() = std::to_string(op(A.i, B.i));
        } else if (isFloat(mCallStack.top(), A.x) && isFloat(str, B.x)) {
            mCallStack.top() = std::to_string(op(A.x, B.x));
        } else {
            throw std::invalid_argument("ab: invalid arguments \"" + mCallStack.top() + "\" and \"" + str + "\"");
        }
    }

    /// @brief apply a boolean test, e.g. a == b, to the two top elements, assuming they are floats, integers or strings
    template <typename T>
    void boolean(T test){
        union {std::int32_t i; float x;} A, B;
        const std::string str = mCallStack.pop();
        if (isInt(mCallStack.top(), A.i) && isInt(str, B.i)) {
            mCallStack.top() = test(A.i, B.i) ? "1" : "0";
        } else if (isFloat(mCallStack.top(), A.x) && isFloat(str, B.x)) {
            mCallStack.top() = test(A.x, B.x) ? "1" : "0";
        } else {// string
            mCallStack.top() = test(mCallStack.top(), str) ? "1" : "0";
        }
    }
};// Processor class

// ==============================================================================================================

/// @brief Abstract base class
struct BaseLoop
{
    BaseLoop(Memory &s, ActIterT i, const std::string &n) : memory(s), begin(i), name(n), pos(0) {}
    virtual ~BaseLoop() {
        memory.clear(name);
        memory.clear("#"+name);
    }
    virtual bool valid() = 0;
    virtual bool next() = 0;
    template <typename T>
    T get() const { return strTo<T>(memory.get(name)); }
    template <typename T>
    void set(T v){
        memory.set(name, v);
        memory.set("#"+name, pos);
    }
    void print(std::ostream& os = std::cerr) const {
        os << "Processing: " << name << " = " << memory.get(name) << ", counter #" << name << " = " << pos <<std::endl;
    }

    Memory&     memory;
    ActIterT    begin;// marks the start of the for loop
    std::string name;
    size_t      pos;// loop counter starting at zero
};// BaseLoop struct

// ==============================================================================================================

template <typename T>
struct ForLoop : public BaseLoop
{
public:

    ForLoop(Memory &s, ActIterT i, const std::string &n, const std::vector<T> &v) : BaseLoop(s, i, n), vec(1) {
        if (v.size()!=2 && v.size()!=3)  throw std::invalid_argument("ForLoop: expected two or three arguments, i=1,9 or i=1,9,2");
        for (size_t i=0; i<v.size(); ++i) vec[i] = v[i];
        if (this->valid()) this->set(vec[0]);
    }
    virtual ~ForLoop() {}
    bool valid() override {return vec[0] < vec[1];}
    bool next() override {
        ++pos;
        vec[0] = this->template get<T>() + vec[2];// read from memory
        if (vec[0] < vec[1]) this->set(vec[0]);
        return vec[0] < vec[1];
    }

private:

    using BaseLoop::pos;
    math::Vec3<T> vec;
};// ForLoop struct

// ==============================================================================================================

class EachLoop : public BaseLoop
{
public:

    EachLoop(Memory &s, ActIterT i, const std::string &n, const VecS &v) : BaseLoop(s,i,n), vec(v.begin(), v.end()){
        if (this->valid()) this->set(vec[0]);
    }
    virtual ~EachLoop() {}
    bool valid() override {return pos < vec.size();}
    bool next() override {
        if (++pos < vec.size()) this->set(vec[pos]);
        return pos < vec.size();
    }

private:

    using BaseLoop::pos;
    const VecS vec;// list of all string values
};// EachLoop class

// ==============================================================================================================

class IfLoop : public BaseLoop
{
public:

    IfLoop(Memory &s, ActIterT i) : BaseLoop(s,i,"") {}
    virtual ~IfLoop() {}
    bool valid() override {return true;}
    bool next() override {return false;}
};// IfLoop class

// ==============================================================================================================

struct Parser {
    Parser(std::vector<Option> &&def);
    void parse(int argc, char *argv[]);
    inline void finalize();
    inline void run();
    inline void setDefaults();
    void print(std::ostream& os = std::cerr) const {for (auto &a : actions) a.print(os);}

    inline std::string getStr(const std::string &name) const;
    template <typename T>
    T get(const std::string &name) const {return strTo<T>(this->getStr(name));}
    template <typename T>
    inline math::Vec3<T> getVec3(const std::string &name, const char* delimiters = "(),") const;
    template <typename T>
    inline std::vector<T> getVec(const std::string &name, const char* delimiters = "(),") const;

    void usage(const VecS &actions, bool brief) const;
    void usage(bool brief) const {for (auto i = std::next(iter);i!=actions.end(); ++i) std::cerr << this->usage(*i, brief);}
    void usage_all(bool brief) const {for (const auto &a : available) std::cerr << this->usage(a, brief);}
    std::string usage(const Action &action, bool brief) const;
    void addAction(std::string &&name, // primary name of the action
                   std::string &&alias, // brief alternative name for action
                   std::string &&doc, // documentation of action
                   std::vector<Option>   &&options, // list of options for the action
                   std::function<void()> &&parse, // callback function called during parsing
                   std::function<void()> &&run,  // callback function to perform the action
                   size_t anonymous = -1)//defines if un-named options are allowed
    {
      available.emplace_back(std::move(name),    std::move(alias), std::move(doc),
                             std::move(options), std::move(parse), std::move(run), anonymous);
    }
    Action& getAction() {return *iter;}
    const Action& getAction() const {return *iter;}
    void printAction() const {if (verbose>1) iter->print();}

    ActListT            available, actions;
    ActIterT            iter;// iterator pointing to the current actions being processed
    std::unordered_map<std::string, ActIterT> hashMap;
    std::list<std::shared_ptr<BaseLoop>> loops;
    std::vector<Option> defaults;
    int                 verbose;
    mutable size_t      counter;// loop counter used to validate matching "-for/each" and "-end" actions
    mutable Processor   processor;// responsible for storing local variables and executing string expressions
};// Parser struct

// ==============================================================================================================

std::string Parser::getStr(const std::string &name) const
{
  for (auto &opt : iter->options) {
      if (opt.name != name) continue;// linear search
      std::string str = opt.value;// deep copy since it might get modified by map
      processor(str);
      return str;
  }
  throw std::invalid_argument(iter->name+": Parser::getStr: no option named \""+name+"\"");
}// Parser::getStr

// ==============================================================================================================

template <>
std::string Parser::get<std::string>(const std::string &name) const {return this->getStr(name);}

// ==============================================================================================================

template <>
std::vector<std::string> Parser::getVec<std::string>(const std::string &name, const char* delimiters) const
{
    return tokenize(this->getStr(name), delimiters);
}// Parser::getVec

// ==============================================================================================================

template <typename T>
std::vector<T> Parser::getVec(const std::string &name, const char* delimiters) const
{
    VecS v = this->getVec<std::string>(name, delimiters);
    std::vector<T> vec(v.size());
    for (size_t i=0; i<v.size(); ++i) vec[i] = strTo<T>(v[i]);
    return vec;
}// Parser::getVec

// ==============================================================================================================

template <typename T>
math::Vec3<T> Parser::getVec3(const std::string &name, const char* delimiters) const
{
    VecS v = this->getVec<std::string>(name, delimiters);
    if (v.size()!=3) throw std::invalid_argument(iter->name+": Parser::getVec3: not a valid input "+name);
    return math::Vec3<T>(strTo<T>(v[0]), strTo<T>(v[1]), strTo<T>(v[2]));
}// Parser::getVec3

// ==============================================================================================================

void Action::setOption(const std::string &str)
{
    const size_t pos = str.find_first_of("={");// since expressions are only evaluated for values and not for names of values, we only search for '=' before expressions, which start with '{'
    if (pos == std::string::npos || str[pos]=='{') {// str has no "=" or it's an expression so append it to the value of the anonymous option
        if (anonymous>=options.size()) throw std::invalid_argument(name+": does not support un-named option \""+str+"\"");
        options[anonymous].append(str);
    } else if (anonymous<options.size() && str.compare(0, pos, options[anonymous].name) == 0) {
        options[anonymous].append(str.substr(pos+1));
    } else {
        for (Option &opt : options) {
            if (opt.name.compare(0, pos, str, 0, pos) != 0) continue;// find first option with partial match
            opt.value = str.substr(pos+1);
            return;// done
        }
        for (Option &opt : options) {
            if (!opt.name.empty()) continue;// find first option with no name
            opt.name  = str.substr(0,pos);
            opt.value = str.substr(pos+1);
            return;// done
        }
        throw std::invalid_argument(name + ": invalid option \"" + str + "\"");
    }
}// Action::setOption

// ==============================================================================================================

void Action::print(std::ostream& os) const
{
    os << "-" << name;
    for (auto &a : options) os << " " << a.name << "=" << a.value;
    os << std::endl;
}// Action::print

// ==============================================================================================================

Parser::Parser(std::vector<Option> &&def)
  : available()// vector of all available actions
  , actions()//   vector of all selected actions
  , iter()// iterator pointing to the current actions being processed
  , hashMap()
  , loops()// list of all for- and each-loops
  , defaults(def)// by default keep is set to false
  , verbose(1)// verbose level is set to 1 my default
  , counter(1)// 1-based global loop counter associated with 'G'
{
    this->addAction(
        "eval", "", "evaluate string expression",
        {{"str", "", "{1:2:+}", "one or more strings to be processed by the stack-oriented programming language. Non-empty string outputs are printed to the terminal"},
         {"help", "", "*|+,-,...", "print a list of all or specified list operations each with brief documentation"}},
        [](){},
        [&](){
            OPENVDB_ASSERT(iter->name == "eval");
            if (!iter->options[1].value.empty()) {
                if (iter->options[1].value=="*") {
                    processor.help();
                } else {
                    processor.help(tokenize(iter->options[1].value, ","));
                }
            }
            std::string str = iter->options[0].value;// copy
            processor(str);// <- evaluate string
            for (auto s : tokenize(str, ",")) std::cerr << s << std::endl;// split and print
        }, 0
    );

    this->addAction(
        "quiet", "", "disable printing to the terminal",{},
        [&](){verbose=0;},[&](){verbose=0;}
    );

    this->addAction(
        "verbose", "", "print timing information to the terminal",{},
        [&](){verbose=1;},[&](){verbose=1;}
    );

    this->addAction(
        "debug", "", "print debugging information to the terminal",{},
        [&](){verbose=2;},[&](){verbose=2;}
    );

    this->addAction(
        "default", "", "define default values to be used by subsequent actions",
        std::vector<Option>(defaults), // using std::move produces error: moving a temporary object prevents copy elision
        [&](){assert(iter->name == "default");
              std::vector<Option> &src = iter->options, &dst = defaults;
              OPENVDB_ASSERT(src.size() == dst.size());
              for (size_t i=0; i<src.size(); ++i) if (!src[i].value.empty()) dst[i].value = src[i].value;},
        [](){}
    );

    // Lambda function used to skip loops by forwarding iterator to matching -end.
    // Note, this function assumes that -for,-each,-if all have a matching -end, which
    // was enforced during parsing by increasing and decreasing "counter" and checking
    // that it never becomes negative and always ends up with a value of zero.
    auto skipToEnd = [](auto &it){
        for (int i = 1; i > 0;) {
            const std::string &name = (++it)->name;
            if (name == "end") {
                i -= 1;
            } else if (name == "for" || name == "each" || name == "if") {
                i += 1;
            }
        }
        OPENVDB_ASSERT(it->name == "end");
    };

    this->addAction(
        "for", "", "start of for-loop over a user-defined loop variable and range.",
        {{"", "", "i=0,9|i=0,9,2", "define name of loop variable and its range."}},
        [&](){++counter;},
        [&](){
            OPENVDB_ASSERT(iter->name == "for");
            const std::string &name = iter->options[0].name;
            std::shared_ptr<BaseLoop> loop;
            try {
                loop=std::make_shared<ForLoop<int>>(processor.memory(), iter, name, this->getVec<int>(name,","));
            } catch (const std::invalid_argument &){
                loop=std::make_shared<ForLoop<float>>(processor.memory(), iter, name, this->getVec<float>(name,","));
            }
            if (loop->valid()) {
                loops.push_back(loop);
                if (verbose) loop->print();
            } else {
                skipToEnd(iter);// skip to matching -end
            }
        }
    );

    this->addAction(
        "each", "", "start of each-loop over a user-defined loop variable and list of values.",
        {{"", "", "s=sphere,bunny,...", "defined name of loop variable and list of its values."}},
        [&](){++counter;},
        [&](){
            OPENVDB_ASSERT(iter->name == "each");
            const std::string &name = iter->options[0].name;
            auto loop = std::make_shared<EachLoop>(processor.memory(), iter, name, this->getVec<std::string>(name,","));
            if (loop->valid()) {
                loops.push_back(loop);
                if (verbose) loop->print();
            } else {
                skipToEnd(iter);// skip to matching -end
            }
        }, 0
    );

    this->addAction(
        "if", "", "start of if-scope. If the value of its option, named test, evaluates to false the entire scope is skipped",
        {{"test", "", "0|1|false|true", "boolean value used to test if-statement"}},
        [&](){++counter;},
        [&](){
            OPENVDB_ASSERT(iter->name == "if");
            if (this->get<bool>("test")) {
                loops.push_back(std::make_shared<IfLoop>(processor.memory(), iter));
            } else {
                skipToEnd(iter);// skip to matching -end
            }
        }, 0
    );

    this->addAction(
        "end", "", "marks the end scope of \"-for,-each,and -if\" control actions", {},
        [&](){
            if (counter<=0) throw std::invalid_argument("Parser: -end must be preceeded by -for,-each, or -if");
            --counter;},
        [&](){
            OPENVDB_ASSERT(iter->name == "end");
            auto loop = loops.back();// current loop
            if (loop->next()) {// rewind loop
                iter = loop->begin;
                if (verbose) loop->print();
            } else {// exit loop
                loops.pop_back();
            }}
    );
}// Parser::Parser

// ==============================================================================================================

void Parser::run()
{
    for (iter=actions.begin();iter!=actions.end();++iter) iter->run();
}// Parser::run(

// ==============================================================================================================

void Parser::finalize()
{
    // sort available actions according to their name
    available.sort([](const Action &a, const Action &b){return a.name < b.name;});

    // build hash table for accelerated random lookup
    for (auto it = available.begin(); it != available.end(); ++it) {
        hashMap.insert({it->name, it});
        if (it->alias!="") hashMap.insert({it->alias, it});
    }
}// Parser::finalize

// ==============================================================================================================

void Parser::parse(int argc, char *argv[])
{
    OPENVDB_ASSERT(!hashMap.empty());
    if (argc <= 1) throw std::invalid_argument("Parser: No arguments provided, try " + getFile(argv[0]) + " -help\"");
    counter = 0;// reset to check for matching {for,each}/end loops
    for (int i=1; i<argc; ++i) {
        const std::string str = argv[i];
        size_t pos = str.find_first_not_of("-");
        if (pos==std::string::npos) throw std::invalid_argument("Parser: expected an action but got \""+str+"\"");
        auto search = hashMap.find(str.substr(pos));//first remove all leading "-"
        if (search != hashMap.end()) {
            actions.push_back(*search->second);// copy construction of Action
            iter = std::prev(actions.end());// important
            while(i+1<argc && argv[i+1][0] != '-') iter->setOption(argv[++i]);
            iter->init();// optional callback function unique to action
        } else {
            throw std::invalid_argument("Parser: unsupported action \""+str+"\"\n");
        }
    }// loop over all input arguments
    if (counter!=0) throw std::invalid_argument("Parser: Unmatched pairing of {-for,-each,-if} and -end");
}// Parser::parse

// ==============================================================================================================

void Parser::usage(const VecS &actions, bool brief) const
{
    for (const std::string &str : actions) {
        auto search = hashMap.find(str);
        if (search == hashMap.end()) throw std::invalid_argument(iter->name+": Parser:::usage: unsupported action \""+str+"\"\n");
        std::cerr << this->usage(*search->second, brief);
    }
}// Parser::usage

// ==============================================================================================================

std::string Parser::usage(const Action &action, bool brief) const
{
    std::stringstream ss;
    const static int w = 17;
    auto op = [&](std::string line, size_t width, bool isSentence) {
        if (isSentence) {
            line[0] = static_cast<char>(std::toupper(line[0]));// capitalizestd::string name, value, example, documentation;');// punctuate
        }
        width += w;
        const VecS words = tokenize(line, " ");
        for (size_t i=0, n=width; i<words.size(); ++i) {
            ss << words[i] << " ";
            n += words[i].size() + 1;
            if (i<words.size()-1 && n > 80) {// exclude last word
                ss << std::endl << std::left << std::setw(static_cast<int>(width)) << "";
                n = width;
            }
        }
        ss << std::endl;
    };

    std::string name = "-" + action.name;
    if (action.alias!="") name += ",-" + action.alias;
    ss << std::endl << std::left << std::setw(w) << name;
    std::string line;
    if (brief) {
        for (auto &opt : action.options) line+=opt.name+(opt.name!=""?"=":"")+opt.example+" ";
        if (line.empty()) line = "This action takes no options.";
        op(line, 0, false);
    } else {
        op(action.documentation, 0, true);
        size_t width = 0;
        for (const auto &opt : action.options) width = std::max(width, opt.name.size());
        width += 4;
        for (const auto &opt : action.options) {
            ss << std::endl << std::setw(w) << "" << std::setw(static_cast<int>(width));
            if (opt.name.empty()) {
                size_t p = opt.example.find('=');
                ss << opt.example.substr(0, p) << opt.example.substr(p+1);
            } else {
                ss << opt.name << opt.example;
            }
            ss << std::endl << std::left << std::setw(w+static_cast<int>(width)) << "";
            op(opt.documentation, width, true);
        }
    }
    return ss.str();
}// Parser::usage

// ==============================================================================================================

void Parser::setDefaults()
{
    for (auto &dst : iter->options) {
        if (dst.value.empty()) {// only set default value if the existing value un-defined?
            for (auto &src : defaults) {
                if (dst.name == src.name) {
                    dst.value = src.value;
                    break;//only breaks the innermost for-loop
                }
            }
        }
    }
}// Parser::setDefaults

// ==============================================================================================================

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_PARSER_HAS_BEEN_INCLUDED
