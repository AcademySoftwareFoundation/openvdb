// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Parser.h
///
/// @brief Defines various classes (Parser, Option, Action, Loop) for processing
///        command-line arguments.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_PARSER_HAS_BEEN_INCLUDED
#define VDB_TOOL_PARSER_HAS_BEEN_INCLUDED

#include <iostream>
#include <sstream>
#include <string> // for std::string, std::stof and std::stoi
#include <algorithm> // std::sort
#include <vector>
#include <list>
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
    inline void append(const std::string &str);
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

struct Loop {
    ActIterT begin;// marks the start of the for loop
    char vName, cName;// one character name of loop variable and loop counter, eg. vName=0,10,1 cName=I
    VecS vec;// list of all string values
    size_t pos;// index of the string value in vec
    Loop(ActIterT it, char a, char b, const VecS &s) : begin(it), vName(a), cName(b), vec(s.begin(), s.end()), pos(0) {}
    Loop(ActIterT it, char a, char b, const VecF &f);
    bool next() { return ++pos < vec.size(); }
    const std::string& value() const {return vec[pos];}
    std::string counter() const {return std::to_string(pos+1);}// 1-based
};// Loop struct

// ==============================================================================================================
struct Parser {
    ActListT            available;
    ActListT            actions;
    ActIterT            action_iter;
    std::unordered_map<std::string, ActIterT> hashMap;
    std::list<Loop>     loops;
    std::vector<Option> defaults;
    int                 verbose;
    mutable size_t      counter;// loop counter

    Parser(std::vector<Option> &&def);
    void parse(int argc, char *argv[]);
    inline void finalize();
    inline void run();
    inline void printLoop();
    inline void beginLoop();
    inline void endLoop();
    inline void updateDefaults();
    inline void setDefaults();
    void print(std::ostream& os = std::cerr) const {for (auto &a : actions) a.print(os); }
    inline void map(std::string &str) const;

    inline std::string getStr(const std::string &name) const;
    bool getBool(const std::string &name) const {return str2bool(this->getStr(name));}
    int getInt(const std::string &name) const {return str2int(this->getStr(name));}
    float getFloat(const std::string &name) const {return str2float(this->getStr(name));}
    double getDouble(const std::string &name) const {return str2double(this->getStr(name));}

    inline Vec3i getVec3I(const std::string &name, const char* delimiters = "(),") const;
    inline Vec3f getVec3F(const std::string &name, const char* delimiters = "(),") const;
    inline Vec3d getVec3D(const std::string &name, const char* delimiters = "(),") const;
    inline VecI  getVecI( const std::string &name, const char* delimiters = "(),") const;
    inline VecF  getVecF( const std::string &name, const char* delimiters = "(),") const;
    inline VecS  getVecS( const std::string &name, const char* delimiters = "(),") const;

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

void Option::append(const std::string &str)
{
    if (value.empty()) {
        value = str;
    } else {
        value += "," + str;
    }
}

// ==============================================================================================================

void Action::setOption(const std::string &str)
{
    const size_t pos = str.find('=');
    if (pos == std::string::npos) {// str has no "=" so append it to the value of the first option
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

Loop::Loop(ActIterT it, char a, char b, const VecF &f)
  : begin(it)
  , vName(a)
  , cName(b)
  , pos(0)
{
    if (f.size()!=3 || f[0]>=f[1]) throw std::invalid_argument("for-loop expected 3 arguments a,b,c where a<b, e.g. i=0,1,1 or f=1.1,2.3,0.1");
    if (floor(f[0]) == f[0] && floorf(f[2]) == f[2]) {// is integer
        for (int i=int(f[0]), d=int(f[2]); i<f[1]; i+=d) vec.push_back(std::to_string(i));
    } else {
        for (float x=f[0]; x<f[1]; x+=f[2]) vec.push_back(std::to_string(x));
    }
}// Loop constructor for for-loops

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
{
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
        "for", "", "beginning of for-loop over a user-defined loop variable and range, e.g. i=0,10,1",
        {{"", "", "i=0,10,1", "define name of loop variable and its range"},
         {"I", "I", "I", "variable name of the counter associated with the for-loop"}},
         [&](){++counter; const auto &opt = action_iter->options;
               if (opt[0].name.size() !=1 || !std::isalpha(opt[0].name[0]))  throw std::invalid_argument("for: expected a single alphabetic character for the loop variable, not "+opt[0].name);
               if (opt[1].value.size()!=1 || !std::isalpha(opt[1].value[0])) throw std::invalid_argument("for: expected a single alphabetic character for the loop counter, not "+opt[1].value);
               if (opt[0].name == opt[1].value) throw std::invalid_argument("for: loop variable and counter cannot be identical");
               if (opt[0].name == "G"|| opt[1].value == "G") throw std::invalid_argument("for: G is reserved for a global counter");},
         [&](){this->beginLoop();}
    );

    this->addAction(
        "each", "", "beginning of each-loop over a user-defined loop variable and list of values, e.g. s=sphere,bunny",
      {{"", "", "s=foo,bar,...", "defined name of loop variable and list of its values"},
       {"I", "J", "J", "variable name of the counter associated with the each-loop"}},
       [&](){++counter; const auto &opt = action_iter->options;
             if (opt[1].value.size()!=1 || !std::isalpha(opt[1].value[0])) throw std::invalid_argument("for: expected I=[A-Z], not "+opt[1].value);
             if (opt[0].name == opt[1].value) throw std::invalid_argument("for: loop variable and counter cannot be identical");
             if (opt[0].name == "G"|| opt[1].value == "G") throw std::invalid_argument("for: G is reserved for a global counter");},
       [&](){this->beginLoop();}
    );

    this->addAction(
        "end", "", "marks the end scope of a for- or each-loop", {},
        [&](){if (counter<=0) throw std::invalid_argument("Parser: -end must be preceeded by -for or -each");
              --counter;}, [&](){this->endLoop();}
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
    if (argc <= 1) throw std::invalid_argument("Parser: No arguments provided, try " + getFileBase(argv[0]) + " -help\"");
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

// E.g. maps "%i" -> "4", %i2" -> "04"  and %G to counter++
void Parser::map(std::string &str) const
{
    const size_t pos = str.find('%');
    if (pos == std::string::npos) return;// early out
    for (auto it=loops.crbegin(); it!=loops.crend() && replace(str, it->vName, it->value()  ,pos)>=0
                                                    && replace(str, it->cName, it->counter(),pos)>=0; ++it);
    while(replace(str, 'G', std::to_string(counter), pos, 1) == 1) ++counter;
}

// ==============================================================================================================

std::string Parser::getStr(const std::string &name) const
{
  for (auto &opt : action_iter->options) {
      if (opt.name != name) continue;
      std::string str = opt.value;// deep copy since it might get modified by map
      this->map(str);
      return str;
  }
  throw std::invalid_argument(action_iter->name+": Parser::getStr: no option named \""+name+"\"");
}

// ==============================================================================================================

Vec3i Parser::getVec3I(const std::string &name, const char* delimiters) const
{
    std::string str = this->getStr(name);
    VecS v = tokenize(str, delimiters);
    if (v.size()!=3) throw std::invalid_argument(action_iter->name+": Parser::getVec3I: not a valid input "+str);
    return Vec3i(str2int(v[0]), str2int(v[1]), str2int(v[2]));
}

// ==============================================================================================================

Vec3f Parser::getVec3F(const std::string &name, const char* delimiters) const
{
    std::string str = this->getStr(name);
    VecS v = tokenize(str, delimiters);
    if (v.size()!=3) throw std::invalid_argument(action_iter->name+": Parser::getVec3F: not a valid input "+str);
    return Vec3f(str2float(v[0]), str2float(v[1]), str2float(v[2]));
}

// ==============================================================================================================

Vec3d Parser::getVec3D(const std::string &name, const char* delimiters) const
{
    std::string str = this->getStr(name);
    VecS v = tokenize(str, delimiters);
    if (v.size()!=3) throw std::invalid_argument(action_iter->name+": Parser::getVec3D: not a valid input "+str);
    return Vec3d(str2double(v[0]), str2double(v[1]), str2double(v[2]));
}

// ==============================================================================================================

std::vector<float> Parser::getVecF(const std::string &name, const char* delimiters) const
{
    std::string str = this->getStr(name);
    VecS v = tokenize(str, delimiters);
    VecF vec(v.size());
    for (int i=0; i<v.size(); ++i) vec[i] = str2float(v[i]);
    return vec;
}

// ==============================================================================================================

std::vector<int> Parser::getVecI(const std::string &name, const char* delimiters) const
{
    std::string str = this->getStr(name);
    VecS v = tokenize(str, delimiters);
    VecI vec(v.size());
    for (int i=0; i<v.size(); ++i) vec[i] = str2int(v[i]);
    return vec;
}

// ==============================================================================================================

std::vector<std::string> Parser::getVecS(const std::string &name, const char* delimiters) const
{
    return tokenize(this->getStr(name), delimiters);
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
    auto op = [&](const std::string &line, size_t width = 0) {
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
    if (brief) {
        std::string line;
        for (auto &opt : action.options) line+=opt.name+(opt.name!=""?"=":"")+opt.example+" ";
        if (line.empty()) line = "this action takes no options";
        op(line);
    } else {
        op(action.documentation);
        size_t width = 0;
        for (const auto &opt : action.options) width = std::max(width, opt.name.size());
        width += 4;
        for (const auto &opt : action.options) {
            ss << std::setw(w) << "" << std::setw(width) << ("\""+opt.name+"\": ");
            op(opt.documentation, width);
        }
    }
    return ss.str();
}

// ==============================================================================================================

void Parser::printLoop()
{
    if (verbose) {
        const Loop& loop = loops.back();
        std::cerr << "\nProcessing " << loop.vName << " = " << loop.value()
                  << " and " << loop.cName << " = " << loop.counter() << "\n";
    }
}

// ==============================================================================================================

void Parser::beginLoop()
{
    assert(findMatch(action_iter->name,{"for", "each"}));
    const char a=action_iter->options[0].name[0], b=action_iter->options[1].value[0];
    for (const Loop &p : loops) {
        if (p.vName==a) throw std::invalid_argument(action_iter->name+": Duplicate loop variable \""+std::string(1,a)+"\"");
        if (p.cName==b) throw std::invalid_argument(action_iter->name+": Duplicate loop variable \""+std::string(1,b)+"\"");
    }
    if (action_iter->name == "for") {
        loops.emplace_back(action_iter, a, b, vectorize<float>(action_iter->options[0].value, ","));
    } else {
        loops.emplace_back(action_iter, a, b, tokenize(action_iter->options[0].value, ","));
    }
    this->printLoop();
}

// ==============================================================================================================

void Parser::endLoop()
{
    auto &loop = loops.back();// current loop
    if (loop.next()) {// rewind loop
        action_iter = loop.begin;
        this->printLoop();
    } else {// exit loop
        loops.pop_back();
    }
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
