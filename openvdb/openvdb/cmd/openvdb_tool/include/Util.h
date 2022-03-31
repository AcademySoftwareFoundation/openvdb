// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @buthor Ken Museth
///
/// @file Util.h
///
/// @brief Utility functions
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_UTIL_HAS_BEEN_INCLUDED
#define VDB_TOOL_UTIL_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip> // for std::setfill
#include <cstring>// for std::strtok
#include <vector>
#include <algorithm>// for std::transform
#include <sys/stat.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

/// @brief return @c true if @b pattern is contained in @b str starting from position @b pos
inline bool contains(const std::string &str, const std::string &pattern, size_t pos = 0)
{
    return str.find(pattern, pos) != std::string::npos;
}

/// @brief return @c true if @b character is contained in @b str starting from position @b pos
inline bool contains(const std::string &str, char character, size_t pos = 0)
{
    return str.find(character, pos) != std::string::npos;
}

/// @brief return @c true if @b fileOrPath names an existing file or path
inline bool file_exists(const std::string &fileOrPath)
{
    struct stat buffer;
    return stat(fileOrPath.c_str(), &buffer) == 0;
}
/// @return the filename, i.e. "base0123.ext" if input is "path/base0123.ext"
inline std::string getFile(const std::string &str)
{
    return str.substr(str.find_last_of("/\\") + 1);
}

    /// @return the path, i.e. "path" if input is "path/base0123.ext"
    inline std::string getPath(const std::string &str)
    {
        const size_t pos = str.find_last_of("/\\");
        return pos >= str.length() ? "." : str.substr(0,pos);
    }

    /// @return the name, i.e. "base0123" if input is "path/base0123.ext"
    inline std::string getName(const std::string &str)
    {
        const size_t start = str.find_last_of("/\\") + 1; // valid offset or npos + 1 = 0
        return str.substr(start, str.find_last_of(".") - start);
    }

    /// @return the base, i.e. "base0123.ext" if input is "path/base0123.ext"
    inline std::string getBase(const std::string &str)
    {
        const std::string name = getName(str);
        return name.substr(0, name.find_last_not_of("0123456789")+1);
    }

    /// @return the file number, i.e. "0123" if input is "path/base0123.ext"
    inline std::string getNumber(const std::string &str)
    {
        const std::string name = getName(str);
        const size_t pos = name.find_first_of("0123456789");
        return name.substr(pos, name.find_last_of("0123456789") + 1 - pos);
    }

    /// @return the file extension, i.e. "ext" if input is "path/base0123.ext"
    inline std::string getExt(const std::string &str)
    {
        const size_t pos = str.find_last_of('.');
        return pos >= str.length() ? "" : str.substr(pos + 1);
    }

    /// @brief Turns all characters in a string into lower case
    inline std::string &to_lower_case(std::string &str)
    {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
        return str;
    }

    /// @brief Turns all characters in a string into upper case
    inline std::string &to_upper_case(std::string &str)
    {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::toupper(c); });
        return str;
    }

    /// @brief Turns all characters in a string into lower case
    inline std::string to_lower_case(const std::string &str)
    {
        std::string tmp = str;
        return to_lower_case(tmp);
    }

    /// @brief Turns all characters in a string into upper case
    inline std::string to_upper_case(const std::string &str)
    {
        std::string tmp = str;
        return to_upper_case(tmp);
    }

    /// @brief return the 1-based index of the first matching word in a string-vector with comma-seperated words.
    /// @details findMatch("ba", {"abc,a", "aab,c,ba"}) returns 2 since "ba" is a word in the second entry of vec
    inline int findMatch(const std::string &word, const std::vector<std::string> &vec)
    {
        for (size_t i = 0; i < vec.size(); ++i){
            size_t p = std::string::npos;
            do {
                ++p;
                const size_t q = vec[i].find(',', p);
                if (vec[i].compare(p, q - p, word) == 0) return i + 1; // 1-based return value
                p = q;
            } while (p != std::string::npos);
        }
        return 0;
    }

    /// @brief return the 1-based on of array on candidate extensions in @b suffix that matches the extension in @b fileName.
    ///        If ignore_case is true the case of the file extension is ignored.
    inline int findFileExt(const std::string &fileName, const std::vector<std::string> &suffix, bool ignore_case = true)
    {
        std::string ext = getExt(fileName);
        if (ignore_case) to_lower_case(ext);
        return findMatch(ext, suffix);
    }

    /// @brief Returns a new string where the leading and training characters of type @b c have been trimmed away.
    inline std::string trim(const std::string &s, std::string c = " \n\r\t\f\v")
    {
        const size_t start = s.find_first_not_of(c), end = s.find_last_not_of(c);
        return start == std::string::npos ? "" : s.substr(start, 1 + end - start);
    }

    inline std::vector<size_t> find_all(const std::string &str, char name = '%')
    {
        std::vector<size_t> vec;
        for (size_t pos = str.find(name); pos != std::string::npos; pos = str.find(name, pos + 1)) vec.push_back(pos);
        return vec;
    }

    /// @brief Returns true if the first substring of @b str equals the character sequence of @b pattern.
    inline bool starts_with(const std::string &str, const std::string &pattern)
    {
        const size_t m = str.length(), n = pattern.length();
        return m >= n ? str.compare(0, n, pattern) == 0 : false;
    }

    /// @brief Returns true if the last substring of @b str equals the character sequence of @b pattern.
    inline bool ends_with(const std::string &str, const std::string &pattern)
    {
        const size_t m = str.length(), n = pattern.length();
        return m >= n ? str.compare(m - n, n, pattern) == 0 : false;
    }

    /// @return return true if the floating-point arguments is an integer
    inline bool is_int(float x) {return floorf(x) == x;}

    /// @brief Returns true if the string contains an integer, in which case @b i contains the value.
    /// @note Leading and trailing whilespaces in the input string are not allowed
    inline bool is_int(const std::string &s, int &i)
    {
        size_t pos = 0;
        try {
            i = stoi(s, &pos);
        } catch (const std::invalid_argument &) {
            return false;
        }
        return pos == s.size();
    }

    /// @brief convert @b str into an integer
    inline int str2int(const std::string &str)
    {
        size_t pos = 0;
        int i;
        try{
            i = stoi(str, &pos); // might throw
        } catch (const std::invalid_argument &) {
            pos = std::string::npos;
        }
        if (pos != str.length()) throw std::invalid_argument("str2int: invalid int \"" + str + "\"");
        return i;
    }

    /// @brief convert @b str into a float
    inline float str2float(const std::string &str)
    {
        size_t pos = 0;
        float v;
        try {
            v = stof(str, &pos); // might throw
        } catch (const std::invalid_argument &){
            pos = std::string::npos;
        }
        if (pos != str.size()) throw std::invalid_argument("str2float: invalid float \"" + str + "\"");
        return v;
    }

    /// @brief convert @b str into a double
    inline double str2double(const std::string &str)
    {
        size_t pos = 0;
        double v;
        try {
            v = stod(str, &pos); // might throw
        } catch (const std::invalid_argument &) {
            pos = std::string::npos;
        }
        if (pos != str.size()) throw std::invalid_argument("str2double: invalid double \"" + str + "\"");
        return v;
    }

    /// @brief convert @b str into a bool
    inline bool str2bool(const std::string &str)
    {
        if (str == "1" || to_lower_case(str) == "true") return true;
        if (str == "0" || to_lower_case(str) == "false") return false;
        throw std::invalid_argument("str2bool: invalid bool \"" + str + "\"");
        return "str2bool: internal error"; // should never happen
    }

    template <typename T>
    inline T str2(const std::string &str);

    template <>
    inline int str2<int>(const std::string &str){return str2int(str);}
    template <>
    inline float str2<float>(const std::string &str){return str2float(str);}
    template <>
    inline double str2<double>(const std::string &str){return str2double(str);}
    template <>
    inline bool str2<bool>(const std::string &str){return str2bool(str);}

    /// @brief Returns true if the string contains a float, in which case @b v contains the value.
    /// @note Leading and trailing whilespaces in the input string are not allowed
    /// @warning In this context integers are a subset of floats so be sure to test for integers first!
    inline bool is_flt(const std::string &s, float &v)
    {
        size_t pos = 0;
        try {
            v = stof(s, &pos);
        } catch (const std::invalid_argument &) {
            return false;
        }
        return pos == s.size();
    }

    /// @brief Returns 1 if the string is an integer, 2 if it's a float and otherwise 0. In the first two cases
    ///        the respective values are updated.
    /// @note Leading and trailing whilespaces in the input string are not allowed
    inline int is_number(const std::string &s, int &i, float &v)
    {
        if (is_int(s, i)) {
            return 1;
        } else if (is_flt(s, v)) {
            return 2;
        }
        return 0;
    }

    // alternative delimiters = " ,[]{}()"
    inline std::vector<std::string> tokenize(const std::string &line, const char *delimiters = " ")
    {
        std::vector<char> buffer(line.c_str(), line.c_str() + line.size() + 1);
        std::vector<std::string> tokens;
        char *token = std::strtok(buffer.data(), delimiters);
        while (token){
            tokens.push_back(token);
            token = std::strtok(nullptr, delimiters);
        }
        return tokens; // move semantics
    }

    template <typename T>
    std::vector<T> vectorize(const std::string &line, const char *delimiters = " ");

    template <>
    std::vector<std::string> vectorize<std::string>(const std::string &line, const char *delimiters)
    {
        return tokenize(line, delimiters);
    }

    template <>
    std::vector<int> vectorize<int>(const std::string &line, const char *delimiters)
    {
        std::vector<char> buffer(line.c_str(), line.c_str() + line.size() + 1);
        std::vector<int> tokens;
        char *token = std::strtok(buffer.data(), delimiters);
        while (token){
            tokens.push_back(str2int(token));
            token = std::strtok(nullptr, delimiters);
        }
        return tokens; // move semantics
    }

    template <>
    std::vector<float> vectorize<float>(const std::string &line, const char *delimiters)
    {
        std::vector<char> buffer(line.c_str(), line.c_str() + line.size() + 1);
        std::vector<float> tokens;
        char *token = std::strtok(buffer.data(), delimiters);
        while (token){
            tokens.push_back(str2float(token));
            token = std::strtok(nullptr, delimiters);
        }
        return tokens; // move semantics
    }

    template <>
    std::vector<bool> vectorize<bool>(const std::string &line, const char *delimiters)
    {
        std::vector<char> buffer(line.c_str(), line.c_str() + line.size() + 1);
        std::vector<bool> tokens;
        char *token = std::strtok(buffer.data(), delimiters);
        while (token){
            tokens.push_back(str2bool(token));
            token = std::strtok(nullptr, delimiters);
        }
        return tokens; // move semantics
    }

    /// @brief find first "option=str" in args and then return "str".
    inline std::string findArg(const std::vector<std::string> &args, const std::string &option)
    {
        const size_t pos = option.length();
        for (const std::string &a : args){
            if (a[pos] != '=') continue;
            if (a.compare(0, pos, option) == 0) return a.substr(pos + 1);
        }
        throw std::invalid_argument(args[0] + ": found no option named \"" + option + "\"");
        return "error in findArg"; // should never happen due to the previous throw
    }

    /// @brief find "option=1,3,6" in args and return std::vector<int>{1,3,6}.
    inline std::vector<int> findIntN(const std::vector<std::string> &args, const std::string &option)
    {
        const auto t = tokenize(findArg(args, option), " ,");
        std::vector<int> v(t.size());
        for (int i = 0; i < t.size(); ++i) v[i] = str2int(t[i]);
        return v; // move semantics
    }

    /// @brief find "option=1.3,-3.1,6.0" in args and return std::vector<float>{1.3f,-3.1f,6.0f}.
    std::vector<float> findFltN(const std::vector<std::string> &args, const std::string &option)
    {
        const auto t = tokenize(findArg(args, option), " ,");
        std::vector<float> v(t.size());
        for (int i = 0; i < t.size(); ++i) v[i] = str2float(t[i]);
        return v; // move semantics
    }

    /// @brief return true if the device on which it is executed uses little-endian bit ordering
    inline bool isLittleEndian()
    {
        const unsigned int tmp = 1;
        return (*(char *)&tmp == 1);
    }

    /// @brief return a pseudo random uuid string.
    ///
    /// @details this function approximates a uuid version 4, variant 1 as detailed
    ///          here https://en.wikipedia.org/wiki/Universally_unique_identifier
    ///
    /// @note A true uuid is based on a random 128 bit number. However, std::random_device
    /// generates a 32 bit seed and std::mt19937_64 generates a random 64 bit number. In other
    /// words this uuid is more prone to collision than a true uuid, though in practice
    /// collisions are still extremely rare with our approximation.
    inline std::string uuid()
    {
        static std::random_device                      seed;// used to obtain a seed for the random number engine
        static std::mt19937_64                         prng(seed());// A Mersenne Twister pseudo-random number generator of 64-bit numbers with a state size of 19937 bits.
        static std::uniform_int_distribution<uint64_t> getHex(0, 15), getVar(8, 11);// random numbers in decimal ranges [0,15] and [8,11]
        std::stringstream ss;
        ss << std::hex;
        for (int i=0; i<15; ++i) ss << getHex(prng);// 16 random hex numbers
        ss << "-" << getVar(prng);// variant 1: random hex number hex in {8,9,a,b}, which maps to the integers {8,9,10,11}
        for (int i=0; i<15; ++i) ss << getHex(prng);// 16 random hex numbers
        return ss.str().insert(8,"-").insert(13,"-4").insert(23,"-");//hardcode version 4
    }

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_UTIL_HAS_BEEN_INCLUDED