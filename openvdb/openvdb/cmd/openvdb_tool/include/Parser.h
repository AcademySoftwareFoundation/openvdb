// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Parser.h
///
/// @brief Defines various classes (Translator, Parser, Option, Action, Loop) for processing
///        command-line arguments.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_PARSER_HAS_BEEN_INCLUDED
#define VDB_TOOL_PARSER_HAS_BEEN_INCLUDED

#include <iostream>
#include <sstream>
#include <string> // for std::string, std::stof and std::stoi
#include <algorithm> // std::sort
#include <functional>
#include <vector>
#include <list>
//#include <stack>
#include <initializer_list>
#include <unordered_map>
#include <iterator>// for std::advance
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>

#include <openvdb/openvdb.h>

#include "Util.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

// ==============================================================================================================

/// @brief This class defines string attributes for options, i.e. arguments for actions
struct Option {
    std::string name, value, example, documentation;
    void append(const std::string &v) {value = value.empty() ? v : value + "," + v;}
};

// ==============================================================================================================
struct Action {
    std::string            name;// primary name of action, eg "read"
    std::string            alias;// alternate name for action, eg "i"
    std::string            documentation;// documentation e.g. "read", "i", "files", "read files"
    size_t                 anonymous;// index of the option to which the value of un-named option will be appended, e.g. files
    std::vector<Option>    options;// e.g. {{"grids", "density,sphere"}, {"files", "path/base.ext"}}
    std::function<void()>  init, run;// callback functions

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
    Action(const Action&) = default;
    void setOption(const std::string &str);
    void print(std::ostream& os = std::cerr) const;
};// Action struct

// ==============================================================================================================

using ActListT = std::list<Action>;
using ActIterT = typename ActListT::iterator;
using VecF = std::vector<float>;// vector of floats
using VecI = std::vector<int>;// vector for integers
using VecS = std::vector<std::string>;// vector of strings

// ==============================================================================================================

/// @brief Class that stores variables accessed by the Parser
class Storage
{
    std::unordered_map<std::string, std::string> mData;
public:
    Storage() {}
    std::string get(const std::string &name){
        auto it = mData.find(name);
        if (it == mData.end()) throw std::invalid_argument("Storrage::get: undefined variable \""+name+"\"");
        return it->second;
    }
    void clear() {mData.clear();}
    void clear(const std::string &name) {mData.erase(name);}
    void set(const std::string &name, const std::string &value){mData[name]=value;}
    void set(const std::string &name, const char *value){mData[name]=value;}
    template <typename T>
    void set(const std::string &name, const T &value){mData[name]=std::to_string(value);}
    void print(std::ostream& os = std::cerr) const {for (auto &d : mData) os << d.first <<"="<<d.second<<std::endl; }
    size_t size() const {return mData.size();}
    bool isSet(const std::string &name) const {return mData.find(name)!=mData.end();}
};// Storage

// ==============================================================================================================

class Stack {
    std::vector<std::string> mData;
public:
    Stack(){mData.reserve(10);}
    Stack(std::initializer_list<std::string> d) : mData(d.begin(), d.end()) {}
    size_t size() const {return mData.size();}
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
    void print(std::ostream& os = std::cerr) const {for (auto &s : mData) os << " " << s;}
};// Stack

// ==============================================================================================================

/// @brief   Implements a light-weight stack-oriented programming language (very loosely) inspired by Forth
/// @details Specifically, it uses Reverse Polish Notation to define operations that are evaluated during
///          paring of the command-line arguments (options to be precise).
class Translator
{
    Stack                                        mStack;
    std::unordered_map<std::string, std::function<void()>> mOps;
    Storage                                     &mStorage;
    template <typename T>
    void set(const T &t) {mStack.top() = std::to_string(t);}
    void set(bool t) {mStack.top() = t ? "1" : "0";}
    template <typename OpT>
    void a(OpT op){
        union {std::int32_t i; float x;} A;
        if (is_int(mStack.top(), A.i)) {
            mStack.top() = std::to_string(op(A.i));
        } else if (is_flt(mStack.top(), A.x)) {
            mStack.top() = std::to_string(op(A.x));
        } else {
            throw std::invalid_argument("a: invalid argument \"" + mStack.top() + "\"");
        }
    }
    template <typename T>
    void ab(T op){
        union {std::int32_t i; float x;} A, B;
        const std::string str = mStack.pop();
        if (is_int(mStack.top(), A.i) && is_int(str, B.i)) {
            mStack.top() = std::to_string(op(A.i, B.i));
        } else if (is_flt(mStack.top(), A.x) && is_flt(str, B.x)) {
            mStack.top() = std::to_string(op(A.x, B.x));
        } else {
            throw std::invalid_argument("ab: invalid arguments \"" + mStack.top() + "\" and \"" + str + "\"");
        }
    }
    template <typename T>
    void boolian(T test){
        union {std::int32_t i; float x;} A, B;
        const std::string str = mStack.pop();
        if (is_int(mStack.top(), A.i) && is_int(str, B.i)) {
            mStack.top() = test(A.i, B.i) ? "1" : "0";
        } else if (is_flt(mStack.top(), A.x) && is_flt(str, B.x)) {
            mStack.top() = test(A.i, B.i) ? "1" : "0";
        } else {// string
            mStack.top() = test(mStack.top(), str) ? "1" : "0";
        }
    }

public:
    /// @brief c-tor
    Translator(Storage &storage) : mStorage(storage)
    {
        mOps["path"]=[&](){mStack.top()=getPath(mStack.top());};// path in path/base0123.ext
        mOps["file"]=[&](){mStack.top()=getFile(mStack.top());};// base0123.ext in path/base0123.ext
        mOps["name"]=[&](){mStack.top()=getName(mStack.top());};// base0123 in path/base0123.ext
        mOps["base"]=[&](){mStack.top()=getBase(mStack.top());};// base in path/base0123.ext
        mOps["number"]=[&](){mStack.top()=getNumber(mStack.top());};// 0123 in path/base0123.ext
        mOps["ext"]=[&](){mStack.top()=getExt(mStack.top());};// ext in path/base0123.ext
        mOps["+"]=[&](){this->ab(std::plus<>());};
        mOps["-"]=[&](){this->ab(std::minus<>());};
        mOps["*"]=[&](){this->ab(std::multiplies<>());};
        mOps["/"]=[&](){this->ab(std::divides<>());};
        mOps["++"]=[&](){this->a([](auto& x){return ++x;});};
        mOps["--"]=[&](){this->a([](auto& x){return --x;});};
        mOps["=="]=[&](){this->boolian(std::equal_to<>());};
        mOps["!="]=[&](){this->boolian(std::not_equal_to<>());};
        mOps["<="]=[&](){this->boolian(std::less_equal<>());};
        mOps[">="]=[&](){this->boolian(std::greater_equal<>());};
        mOps["<"]=[&](){this->boolian(std::less<>());};
        mOps[">"]=[&](){this->boolian(std::greater<>());};
        mOps["!"]=[&](){this->set(!str2bool(mStack.top()));};
        mOps["|"]=[&](){bool b=str2bool(mStack.pop());this->set(str2bool(mStack.top())||b);};
        mOps["&"]=[&](){bool b=str2bool(mStack.pop());this->set(str2bool(mStack.top())&&b);};
        mOps["abs"]=[&](){this->a([](auto& x){return math::Abs(x);});};
        mOps["ceil"]=[&](){this->a([](auto& x){return std::ceil(x);});};
        mOps["floor"]=[&](){this->a([](auto& x){return std::floor(x);});};
        mOps["pow2"]=[&](){this->a([](auto& x){return math::Pow2(x);});};
        mOps["pow3"]=[&](){this->a([](auto& x){return math::Pow3(x);});};
        mOps["pow"]=[&](){this->ab([](auto& a, auto& b){return math::Pow(a, b);});};
        mOps["min"]=[&](){this->ab([](auto& a, auto& b){return std::min(a, b);});};
        mOps["max"]=[&](){this->ab([](auto& a, auto& b){return std::max(a, b);});};
        mOps["neg"]=[&](){this->a([](auto& x){return -x;});};
        mOps["sin"]=[&](){this->set(std::sin(str2float(mStack.top())));};
        mOps["cos"]=[&](){this->set(std::cos(str2float(mStack.top())));};
        mOps["tan"]=[&](){this->set(std::tan(str2float(mStack.top())));};
        mOps["asin"]=[&](){this->set(std::asin(str2float(mStack.top())));};
        mOps["acos"]=[&](){this->set(std::acos(str2float(mStack.top())));};
        mOps["atan"]=[&](){this->set(std::atan(str2float(mStack.top())));};
        mOps["r2d"]=[&](){this->set(180.0f*str2float(mStack.top())/math::pi<float>());};// radian to degree
        mOps["d2r"]=[&](){this->set(math::pi<float>()*str2float(mStack.top())/180.0f);};// degree to radian
        mOps["inv"]=[&](){this->set(1.0f/str2float(mStack.top()));};
        mOps["exp"]=[&](){this->set(std::exp(str2float(mStack.top())));};
        mOps["ln"]=[&](){this->set(std::log(str2float(mStack.top())));};
        mOps["log"]=[&](){this->set(std::log10(str2float(mStack.top())));};
        mOps["sqrt"]=[&](){this->set(std::sqrt(str2float(mStack.top())));};
        mOps["int"]=[&](){this->set(int(str2float(mStack.top())));};
        mOps["float"]=[&](){this->set(str2float(mStack.top()));};
        mOps["lower"]=[&](){to_lower_case(mStack.top());};
        mOps["upper"]=[&](){to_upper_case(mStack.top());};
        mOps["dup"]=[&](){mStack.dup();};// x -- x x (push top onto stack)
        mOps["nip"]=[&](){mStack.nip();};// y x -- x (remove the entry below the top)
        mOps["drop"]=[&](){mStack.drop();};// y x -- y (pop the top)
        mOps["swap"]=[&](){mStack.swap();};// y x -- x y (swap the two top entries)
        mOps["over"]=[&](){mStack.over();};// y x -- y x y (push second entry onto top)
        mOps["rot"]=[&](){mStack.rot();};// z y x -- y x z (rotation left)
        mOps["tuck"]=[&](){mStack.tuck();};// z y x -- x z y (rotation right)
        mOps["scrape"]=[&](){mStack.scrape();};// ... x -- x (removed everything except for the top)
        mOps["clear"]=[&](){mStack.clear();};// ... -- (remove everything)
        mOps["squash"]=[&](){// "x" "y" "z" --- "x y z"   combines entire stack into the top
            if (mStack.empty()) return;
            std::stringstream ss;
            mStack.print(ss);
            mStack.scrape();
            mStack.top()=ss.str();
        };
        mOps["replace"]=[&](){// e.g. "foo bar" " _" -- "foo_bar" (replace character in string)
          const std::string old_new = mStack.pop();
          if (old_new.size()!=2) throw std::invalid_argument("Translator::replace: expected two characters, not \""+old_new+"\"");
          std::replace(mStack.top().begin(), mStack.top().end(), old_new[0], old_new[1]);
        };
        mOps["exists"]=[&](){this->set(mStorage.isSet(mStack.top()));};
        mOps["size"]=[&](){mStack.push(std::to_string(mStack.size()));};// push size of stack onto the stack
        mOps["pad0"]=[&](){// add zero-padding of a specified with to a string, e.g. {12:4:pad0} -> 0012
            const int w = str2int(mStack.pop());
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(w) << mStack.top();
            mStack.top() = ss.str();
        };
        mStorage.set("pi", math::pi<float>());
    }
    /// @brief process the specified string
    void operator()(std::string &str)
    {
        for (size_t pos = str.find_first_of("{}"); pos != std::string::npos; pos = str.find_first_of("{}", pos)) {
            if (str[pos]=='}') throw std::invalid_argument("Translator(): expected \"{\" before \"}\" in \""+str.substr(pos)+"\"");
            size_t end = str.find_first_of("{}", pos + 1);
            if (end == std::string::npos || str[end]=='{') throw std::invalid_argument("Translator(): missing \"}\" in \""+str.substr(pos)+"\"");
            for (size_t p=str.find_first_of(":}",pos+1), q=pos+1; p<=end; q=p+1, p=str.find_first_of(":}",q)) {
                if (p == q) {// ignores {:} and {::}
                    continue;
                } else if (str[q]=='$') {// get value
                    mStack.push(mStorage.get(str.substr(q + 1, p - q - 1)));
                } else if (str[q]=='@') {// set value
                    if (mStack.empty()) throw std::invalid_argument("Translator::(): cannot evaluate \""+str.substr(q,p-q)+"\" when the stack is empty");
                    mStorage.set(str.substr(q + 1, p - q - 1), mStack.pop());
                } else if (str.compare(q,3,"if(")==0) {// if-statement: 0|1:if(a) or 0|1:if(a?b)}
                    const size_t i = str.find_first_of("(){}", q+3);
                    if (str[i]!=')') throw std::invalid_argument("Translator():: missing \")\" in if-statement \""+str.substr(q)+"\"");
                    const auto v = tokenize(str.substr(q+3, i-q-3), "?");
                    if (v.size() == 1) {
                        if (str2bool(mStack.pop())) {
                            str.replace(q, i - q + 1, v[0]);
                        } else {
                            str.erase(q - 1, i - q + 2);// also erase the leading ':' character
                        }
                    } else if (v.size() == 2) {
                        str.replace(q, i - q + 1, v[str2bool(mStack.pop()) ? 0 : 1]);
                    } else {
                        throw std::invalid_argument("Translator():: invalid if-statement \""+str.substr(q)+"\"");
                    }
                    end = str.find('}', pos + 1);// needs to be recomputed since str was modified
                    p = q - 1;// rewind
                } else if (str.compare(q,4,"quit")==0) {// quit
                    break;
                } else if (str.compare(q,7,"switch(")==0) {//switch-statement: $1:switch(a:case_a?b:case_b?c:case_c)
                    const size_t i = str.find_first_of("(){}", q+7);
                    if (str[i]!=')') throw std::invalid_argument("Translator():: missing \")\" in switch-statement \""+str.substr(q)+"\"");
                    for (auto s : tokenize(str.substr(q+7, i-q-7), "?")) {
                        const size_t j = s.find(':');
                        if (j==std::string::npos) throw std::invalid_argument("Translator():: missing \":\" in switch-statement \""+str.substr(q)+"\"");
                        if (mStack.top() == s.substr(0,j)) {
                            str.replace(q, i - q + 1, s.substr(j + 1));
                            end = str.find('}', pos + 1);// needs to be recomputed since str was modified
                            p = q - 1;// rewind
                            mStack.drop();
                            break;
                        }
                    }
                    if (str.compare(q,7,"switch(")==0) throw std::invalid_argument("Translator():: no match in switch-statement \""+str.substr(q)+"\"");
                } else {
                    const std::string s = str.substr(q, p - q);
                    auto it = mOps.find(s);
                    if (it != mOps.end()) {
                        it->second();// callback
                    } else {
                        mStack.push(s);
                    }
                }
            }
            if (mStack.empty()) {
                str.erase(pos, end-pos+1);
            } else if (mStack.size()==1) {
                str.replace(pos, end-pos+1, mStack.pop());
            } else {
                std::stringstream ss;
                mStack.print(ss);
                throw std::invalid_argument("Translator::(): compute stack contains more than one entry: " + ss.str());
            }
        }
    }
    std::string operator()(const std::string &_str)
    {
        std::string str = _str;// copy
        (*this)(str);
        return str;
    }
};// Translator

// ==============================================================================================================

/// @brief Abstract base class
struct BaseLoop
{
    Storage&    storage;
    ActIterT    begin;// marks the start of the for loop
    std::string name;
    size_t      pos;// index of the string value in vec
    BaseLoop(Storage &s, ActIterT i, const std::string &n) : storage(s), begin(i), name(n), pos(0) {}
    virtual ~BaseLoop() {storage.clear(name); storage.clear("#"+name);}
    virtual bool next() = 0;
    template <typename T>
    T get() const { return str2<T>(storage.get(name)); }
    template <typename T>
    void set(T v){
        storage.set(name, v);
        storage.set("#"+name, pos);
    }
    void print(std::ostream& os = std::cerr) const { os << "Processing: " << name << " = " << storage.get(name) << " counter=" << pos;}
};

// ==============================================================================================================

template <typename T>
struct ForLoop : public BaseLoop
{
    using BaseLoop::pos;
    math::Vec3<T> vec;
public:
    ForLoop(Storage &s, ActIterT i, const std::string &n, const math::Vec3<T> &v) : BaseLoop(s, i, n), vec(v) {
        if (vec[0]>=vec[1]) throw std::invalid_argument("ForLoop: expected 3 arguments a,b,c where a<b, e.g. i=0,1,1 or f=1.1,2.3,0.1");
        this->set(vec[0]);
    }
    virtual ~ForLoop() {}
    bool next() override {
        ++pos;
        vec[0] = this->template get<T>() + vec[2];// read from storage
        if (vec[0] < vec[1]) this->set(vec[0]);
        return vec[0] < vec[1];
    }
};// ForLoop

// ==============================================================================================================

class EachLoop : public BaseLoop
{
    using BaseLoop::pos;
    const VecS vec;// list of all string values
public:
    EachLoop(Storage &s, ActIterT i, const std::string &n, const VecS &v) : BaseLoop(s,i,n), vec(v.begin(), v.end()){
        if (vec.empty()) throw std::invalid_argument("EachLoop: -each does not accept an empty list");
        this->set(vec[0]);
    }
    virtual ~EachLoop() {}
    bool next() override {
        if (++pos < vec.size()) this->set(vec[pos]);
        return pos < vec.size();
    }
};// EachLoop

// ==============================================================================================================

struct Parser {
    ActListT            available, actions;
    ActIterT            action_iter;
    std::unordered_map<std::string, ActIterT> hashMap;
    std::list<std::shared_ptr<BaseLoop>> loops;
    std::vector<Option> defaults;
    int                 verbose;
    mutable size_t      counter;// loop counter
    mutable Storage     storage;
    mutable Translator  translator;

    Parser(std::vector<Option> &&def);
    void parse(int argc, char *argv[]);
    inline void finalize();
    inline void run();
    inline void updateDefaults();
    inline void setDefaults();
    void print(std::ostream& os = std::cerr) const {for (auto &a : actions) a.print(os);}

    inline std::string getStr(const std::string &name) const;
    template <typename T>
    T get(const std::string &name) const {return str2<T>(this->getStr(name));}
    template <typename T>
    inline math::Vec3<T> getVec3(const std::string &name, const char* delimiters = "(),") const;
    template <typename T>
    inline std::vector<T> getVec(const std::string &name, const char* delimiters = "(),") const;

    void usage(const VecS &actions, bool brief) const;
    void usage(bool brief) const {for (auto i = std::next(action_iter);i!=actions.end(); ++i) std::cerr << this->usage(*i, brief);}
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
    Action& getAction() {return *action_iter;}
    const Action& getAction() const {return *action_iter;}
    void printAction() const {if (verbose>1) action_iter->print();}
};// Parser struct

// ==============================================================================================================

std::string Parser::getStr(const std::string &name) const
{
  for (auto &opt : action_iter->options) {
      if (opt.name != name) continue;// linear search
      std::string str = opt.value;// deep copy since it might get modified by map
      translator(str);
      return str;
  }
  throw std::invalid_argument(action_iter->name+": Parser::getStr: no option named \""+name+"\"");
}

// ==============================================================================================================

template <>
std::string Parser::get<std::string>(const std::string &name) const {return this->getStr(name);}

// ==============================================================================================================

template <>
std::vector<std::string> Parser::getVec<std::string>(const std::string &name, const char* delimiters) const
{
    return tokenize(this->getStr(name), delimiters);
}

// ==============================================================================================================

template <typename T>
std::vector<T> Parser::getVec(const std::string &name, const char* delimiters) const
{
    VecS v = this->getVec<std::string>(name, delimiters);
    std::vector<T> vec(v.size());
    for (int i=0; i<v.size(); ++i) vec[i] = str2<T>(v[i]);
    return vec;
}

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
}

// ==============================================================================================================

void Action::print(std::ostream& os) const
{
    os << "-" << name;
    for (auto &a : options) os << " " << a.name << "=" << a.value;
    os << std::endl;
}

// ==============================================================================================================

Parser::Parser(std::vector<Option> &&def)
  : available()// vector of all available actions
  , actions()//   vector of all selected actions
  , action_iter()// iterator the the current actions being processed
  , hashMap()
  , loops()// list of all for- and each-loops
  , verbose(1)// verbose level is set to 1 my default
  , defaults(def)// by default keep is set to false
  , counter(1)// 1-based global loop counter associated with 'G'
  , translator(storage)
{
    this->addAction(
        "eval", "", "evaluate string expression",
        {{"str", "", "{1:@G}", "one or more strings to be processed by the stack-oriented programming language. Non-empty string outputs are printed to the terminal"}},
        [](){},[&](){
            assert(action_iter->name == "eval");
            std::string str =action_iter->options[0].value;
            translator(str);
            for (auto s : tokenize(str, ",")) std::cerr << s << std::endl;
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
        std::move(std::vector<Option>(defaults)),// move a deep copy
        [&](){this->updateDefaults();}, [](){}
    );

    this->addAction(
        "for", "", "beginning of for-loop over a user-defined loop variable and range.",
        {{"", "", "i=0,10,1", "define name of loop variable and its range."}},
        [&](){++counter;}, [&](){
            assert(action_iter->name == "for");
            const std::string &name = action_iter->options[0].name;
            try {
                loops.push_back(std::make_shared<ForLoop<int>>(storage, action_iter, name, this->getVec3<int>(name,",")));
            } catch (const std::invalid_argument &){
                loops.push_back(std::make_shared<ForLoop<float>>(storage, action_iter, name, this->getVec3<float>(name,",")));
            }
            if (verbose) loops.back()->print();
        }
    );

    this->addAction(
        "each", "", "beginning of each-loop over a user-defined loop variable and list of values.",
      {{"", "", "s=sphere,bunny,...", "defined name of loop variable and list of its values."}},
        [&](){++counter;}, [&](){
            assert(action_iter->name == "each");
            const std::string &name = action_iter->options[0].name;
            loops.push_back(std::make_shared<EachLoop>(storage, action_iter, name, this->getVec<std::string>(name,",")));
            if (verbose) loops.back()->print();
        }, 0
    );

    this->addAction(
        "end", "", "marks the end scope of a for- or each-loop", {},
        [&](){if (counter<=0) throw std::invalid_argument("Parser: -end must be preceeded by -for or -each");
            --counter;},
        [&](){
            assert(action_iter->name == "end");
            auto loop = loops.back();// current loop
            if (loop->next()) {// rewind loop
                action_iter = loop->begin;
                if (verbose) loop->print();
            } else {// exit loop
                loops.pop_back();
            }}
    );
}

// ==============================================================================================================

void Parser::run()
{
    counter=1;// reset for the global counter "%G"
    for (action_iter=actions.begin(); action_iter!=actions.end(); ++action_iter) {
      action_iter->run();
    }
}

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
    //std::cerr << "buckets = " << hashMap.bucket_count() << ", size = " << hashMap.size() << std::endl;
}

// ==============================================================================================================

void Parser::parse(int argc, char *argv[])
{
    assert(!hashMap.empty());
    if (argc <= 1) throw std::invalid_argument("Parser: No arguments provided, try " + getFile(argv[0]) + " -help\"");
    counter = 0;// reset to check for matching {for,each}/end loops
    for (int i=1; i<argc; ++i) {
        const std::string str = argv[i];
        size_t pos = str.find_first_not_of("-");
        if (pos==std::string::npos) throw std::invalid_argument("Parser: expected an action but got \""+str+"\"");
        auto search = hashMap.find(str.substr(pos));//first remove all leading "-"
        if (search != hashMap.end()) {
            actions.push_back(*search->second);// copy construction of Action
            action_iter = std::prev(actions.end());// important
            while(i+1<argc && argv[i+1][0] != '-') action_iter->setOption(argv[++i]);
            action_iter->init();// optional callback function unique to action
        } else {
            throw std::invalid_argument("Parser: unsupported action \""+str+"\"\n");
        }
    }// loop over all input arguments
    if (counter!=0) throw std::invalid_argument("Parser: Unmatched pair of -for/-each and -end");
}

// ==============================================================================================================

template <typename T>
math::Vec3<T> Parser::getVec3(const std::string &name, const char* delimiters) const
{
    VecS v = this->getVec<std::string>(name, delimiters);
    if (v.size()!=3) throw std::invalid_argument(action_iter->name+": Parser::getVec3: not a valid input "+name);
    return math::Vec3<T>(str2<T>(v[0]), str2<T>(v[1]), str2<T>(v[2]));
}

// ==============================================================================================================

void Parser::usage(const VecS &actions, bool brief) const
{
    for (const std::string &str : actions) {
        auto search = hashMap.find(str);
        if (search == hashMap.end()) throw std::invalid_argument(action_iter->name+": Parser:::usage: unsupported action \""+str+"\"\n");
        std::cerr << this->usage(*search->second, brief);
    }
}

// ==============================================================================================================

std::string Parser::usage(const Action &action, bool brief) const
{
    std::stringstream ss;
    const static int w = 17;
    auto op = [&](std::string line, size_t width, bool isSentence) {
        if (isSentence) {
            line[0] = std::toupper(line[0]);// capitalize
            if (line.back()!='.') line.append(1,'.');// punctuate
        }
        width += w;
        const VecS words = tokenize(line, " ");
        for (size_t i=0, n=width; i<words.size(); ++i) {
            ss << words[i] << " ";
            n += words[i].size() + 1;
            if (i<words.size()-1 && n > 80) {// exclude last word
                ss << std::endl << std::left << std::setw(width) << "";
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
            ss << std::endl << std::setw(w) << "" << std::setw(width);
            if (opt.name.empty()) {
                size_t p = opt.example.find('=');
                ss << opt.example.substr(0, p) << opt.example.substr(p+1);
            } else {
                ss << opt.name << opt.example;
            }
            ss << std::endl << std::left << std::setw(w+width) << "";
            op(opt.documentation, width, true);
        }
    }
    return ss.str();
}

// ==============================================================================================================

void Parser::updateDefaults()
{
    assert(action_iter->name == "default");
    const std::vector<Option> &other = action_iter->options;
    assert(defaults.size() == other.size());
    for (int i=0; i<defaults.size(); ++i) {
        if (!other[i].value.empty()) defaults[i].value = other[i].value;
    }
}

// ==============================================================================================================

void Parser::setDefaults()
{
    for (auto &dst : action_iter->options) {
        if (dst.value.empty()) {
            for (auto &src : defaults) {
                if (dst.name == src.name) {
                    dst.value = src.value;
                    break;
                }
            }
        }
    }
}

// ==============================================================================================================

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_PARSER_HAS_BEEN_INCLUDED
