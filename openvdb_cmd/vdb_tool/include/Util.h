// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Util.h
///
/// @brief Utility functions for vdb_tool
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_UTIL_HAS_BEEN_INCLUDED
#define VDB_TOOL_UTIL_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <algorithm>// for std::transform
#include <cctype> // for std::tolower
#include <cstring>// for std::strtok
#include <iomanip> // for std::setfill
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <chrono>
#include <ctime>

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
inline bool fileExists(const std::string &fileOrPath)
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

/// @return the file name with new path, i.e. "path/base0123.ext" if input is "tmp/base0123.ext"
inline std::string replacePath(const std::string &str, const std::string &path)
{
    return path + "/" + getFile(str);
}

/// @return the name, i.e. "base0123" if input is "path/base0123.ext"
inline std::string getName(const std::string &str)
{
    const size_t start = str.find_last_of("/\\") + 1; // valid offset or npos + 1 = 0
    return str.substr(start, str.find_last_of(".") - start);
}

/// @return the base, i.e. "base" if input is "path/base0123.ext"
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

/// @return the file name with a new extension, i.e. "path/base0123.abc" if input is "path/base0123.ext", "abc"
inline std::string replaceExt(const std::string &str, const std::string &ext)
{
    const size_t pos = str.find_last_of('.');
    return (pos >= str.length() ? str + "." : str.substr(0, pos + 1)) + ext;
}

/// @brief Turns all characters in a string into lower case
inline std::string &toLowerCase(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
    return str;
}

/// @brief Turns all characters in a string into upper case
inline std::string &toUpperCase(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::toupper(c); });
    return str;
}

/// @brief Turns all characters in a string into lower case
inline std::string toLowerCase(const std::string &str)
{
    std::string tmp = str;
    return toLowerCase(tmp);
}

/// @brief Turns all characters in a string into upper case
inline std::string toUpperCase(const std::string &str)
{
    std::string tmp = str;
    return toUpperCase(tmp);
}

/// @brief return the 1-based index of the first matching word in a string-vector with comma-separated words.
/// @details findMatch("ba", {"abc,a", "aab,c,ba"}) returns 2 since "ba" is a word in the second entry of vec
inline int findMatch(const std::string &word, const std::vector<std::string> &vec)
{
    for (size_t i = 0; i < vec.size(); ++i){
        size_t p = std::string::npos;
        do {
            ++p;
            const size_t q = vec[i].find(',', p);
            if (vec[i].compare(p, q - p, word) == 0) return static_cast<int>(i + 1); // 1-based return value
            p = q;
        } while (p != std::string::npos);
    }
    return 0;
}

/// @brief Return the 1-based index of the candidate extension in @b suffix that matches the extension of @b fileName.
/// @param fileName    File path whose extension will be tested.
/// @param suffix      Vector of candidate extensions (without leading dot).
/// @param ignore_case If true (the default), the comparison is case-insensitive.
/// @return 1-based index of the matching entry in @b suffix, or 0 if no match.
inline int findFileExt(const std::string &fileName, const std::vector<std::string> &suffix, bool ignore_case = true)
{
    std::string ext = getExt(fileName);
    if (ignore_case) toLowerCase(ext);
    return findMatch(ext, suffix);
}

/// @brief Returns a new string where the leading and trailing characters in @b c have been trimmed away.
/// @param s Input string (unchanged).
/// @param c Set of characters to treat as whitespace (defaults to standard whitespace).
inline std::string trim(const std::string &s, std::string c = " \n\r\t\f\v")
{
    const size_t start = s.find_first_not_of(c), end = s.find_last_not_of(c);
    return start == std::string::npos ? "" : s.substr(start, 1 + end - start);
}

/// @brief Find every occurrence of a character in a string.
/// @param str  String to scan.
/// @param name Character to search for (defaults to '%').
/// @return Vector of zero-based positions of every match.
inline std::vector<size_t> findAll(const std::string &str, char name = '%')
{
    std::vector<size_t> vec;
    for (size_t pos = str.find(name); pos != std::string::npos; pos = str.find(name, pos + 1)) vec.push_back(pos);
    return vec;
}

/// @brief Returns true if the first substring of @b str equals the character sequence of @b pattern.
inline bool startsWith(const std::string &str, const std::string &pattern)
{
    const size_t m = str.length(), n = pattern.length();
    return m >= n ? str.compare(0, n, pattern) == 0 : false;
}

/// @brief Returns true if the last substring of @b str equals the character sequence of @b pattern.
inline bool endsWith(const std::string &str, const std::string &pattern)
{
    const size_t m = str.length(), n = pattern.length();
    return m >= n ? str.compare(m - n, n, pattern) == 0 : false;
}

/// @brief Returns true if the floating-point argument has no fractional part.
inline bool isInt(float x) {return floorf(x) == x;}

/// @brief Returns true if the string contains an integer, in which case @b i contains the value.
/// @note Leading and trailing whitespaces in the input string are not allowed
inline bool isInt(const std::string &s, int &i)
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
inline int strToInt(const std::string &str)
{
    size_t pos = 0;
    int i = 0;
    try{
        i = stoi(str, &pos); // might throw
    } catch (const std::invalid_argument &) {
        pos = std::string::npos;
    }
    if (pos != str.length()) throw std::invalid_argument("strToInt: invalid int \"" + str + "\"");
    return i;
}

/// @brief convert @b str into a float
inline float strToFloat(const std::string &str)
{
    size_t pos = 0;
    float v = 0.f;
    try {
        v = stof(str, &pos); // might throw
    } catch (const std::invalid_argument &){
        pos = std::string::npos;
    }
    if (pos != str.size()) throw std::invalid_argument("strToFloat: invalid float \"" + str + "\"");
    return v;
}

/// @brief convert @b str into a double
inline double strToDouble(const std::string &str)
{
    size_t pos = 0;
    double v = 0.0;
    try {
        v = stod(str, &pos); // might throw
    } catch (const std::invalid_argument &) {
        pos = std::string::npos;
    }
    if (pos != str.size()) throw std::invalid_argument("strToDouble: invalid double \"" + str + "\"");
    return v;
}

/// @brief convert @b str into a bool
inline bool strToBool(const std::string &str)
{
    if (str == "1" || toLowerCase(str) == "true") return true;
    if (str == "0" || toLowerCase(str) == "false") return false;
    throw std::invalid_argument("strToBool: invalid bool \"" + str + "\"");
    return false; // "strToBool: internal error" should never happen
}

/// @brief converts a string of byte size to integer byte size, e.g. "2KB" -> 2048
inline uint64_t strSizeToByteSize(const std::string &str)
{
    const size_t first = str.find_first_not_of(" \t"), last = str.find_last_of("0123456789");
    uint64_t size = static_cast<uint64_t>( strToInt(str.substr(first, last + 1 - first)) );// might throw
    const std::string unit = str.substr(last + 1, str.find_last_not_of(" \t") - last);
    if (unit == "KB") {
        size <<= 10;
    } else if (unit == "MB") {
        size <<= 20;
    } else if (unit == "GB") {
        size <<= 30;
    } else if (unit == "TB") {
        size <<= 40;
    } else if (unit != "B" && unit != "") {
        throw std::invalid_argument("strSizeToByteSize: unsupported unit \"" + unit + "\"");
    }
    return size;
}

/// @brief Generic string-to-T conversion; specialized for int, float, double, and bool.
/// @tparam T Target type. Only int, float, double, and bool are provided.
/// @throw std::invalid_argument if @b str cannot be parsed as a T.
template <typename T>
inline T strTo(const std::string &str);

/// @brief Specialization of strTo for int.
template <>
inline int strTo<int>(const std::string &str){return strToInt(str);}
/// @brief Specialization of strTo for float.
template <>
inline float strTo<float>(const std::string &str){return strToFloat(str);}
/// @brief Specialization of strTo for double.
template <>
inline double strTo<double>(const std::string &str){return strToDouble(str);}
/// @brief Specialization of strTo for bool.
template <>
inline bool strTo<bool>(const std::string &str){return strToBool(str);}

/// @brief Returns true if the string contains a float, in which case @b v contains the value.
/// @note Leading and trailing whitespaces in the input string are not allowed
/// @warning In this context integers are a subset of floats so be sure to test for integers first!
inline bool isFloat(const std::string &s, float &v)
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
/// @note Leading and trailing whitespaces in the input string are not allowed
inline int isNumber(const std::string &s, int &i, float &v)
{
    if (isInt(s, i)) {
        return 1;
    } else if (isFloat(s, v)) {
        return 2;
    }
    return 0;
}

/// @brief Split a string into tokens using any character in @b delimiters as a separator.
/// @param line       Input string to split.
/// @param delimiters Null-terminated set of delimiter characters (e.g. " ,[]{}()").
/// @return Vector of token strings; empty tokens between consecutive delimiters are skipped.
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

/// @brief Tokenize @b line and convert each token to type @c T.
/// @tparam T Target element type. Specializations are provided for std::string, int, float, and bool.
/// @throw std::invalid_argument if any token fails to parse as a T.
template <typename T>
std::vector<T> vectorize(const std::string &line, const char *delimiters = " ");

/// @brief Specialization of vectorize for std::string (identical to tokenize).
template <>
std::vector<std::string> vectorize<std::string>(const std::string &line, const char *delimiters)
{
    return tokenize(line, delimiters);
}

/// @brief Specialization of vectorize for int.
template <>
std::vector<int> vectorize<int>(const std::string &line, const char *delimiters)
{
    std::vector<char> buffer(line.c_str(), line.c_str() + line.size() + 1);
    std::vector<int> tokens;
    char *token = std::strtok(buffer.data(), delimiters);
    while (token){
        tokens.push_back(strToInt(token));
        token = std::strtok(nullptr, delimiters);
    }
    return tokens; // move semantics
}

/// @brief Specialization of vectorize for float.
template <>
std::vector<float> vectorize<float>(const std::string &line, const char *delimiters)
{
    std::vector<char> buffer(line.c_str(), line.c_str() + line.size() + 1);
    std::vector<float> tokens;
    char *token = std::strtok(buffer.data(), delimiters);
    while (token){
        tokens.push_back(strToFloat(token));
        token = std::strtok(nullptr, delimiters);
    }
    return tokens; // move semantics
}

/// @brief Specialization of vectorize for bool (accepts "0/1" or "true/false").
template <>
std::vector<bool> vectorize<bool>(const std::string &line, const char *delimiters)
{
    std::vector<char> buffer(line.c_str(), line.c_str() + line.size() + 1);
    std::vector<bool> tokens;
    char *token = std::strtok(buffer.data(), delimiters);
    while (token){
        tokens.push_back(strToBool(token));
        token = std::strtok(nullptr, delimiters);
    }
    return tokens; // move semantics
}

/// @brief Find the first argument of the form "option=value" in @b args and return "value".
/// @param args   Argument vector to search.
/// @param option Option name to match exactly (no partial matching).
/// @throw std::invalid_argument if no argument with the requested option name is found.
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

/// @brief Find "option=1,3,6" in @b args and return std::vector<int>{1,3,6}.
/// @throw std::invalid_argument if @b option is missing or any token fails to parse as an int.
inline std::vector<int> findIntN(const std::vector<std::string> &args, const std::string &option)
{
    const auto t = tokenize(findArg(args, option), " ,");
    std::vector<int> v(t.size());
    for (size_t i = 0; i < t.size(); ++i) v[i] = strToInt(t[i]);
    return v; // move semantics
}

/// @brief Find "option=1.3,-3.1,6.0" in @b args and return std::vector<float>{1.3f,-3.1f,6.0f}.
/// @throw std::invalid_argument if @b option is missing or any token fails to parse as a float.
std::vector<float> findFltN(const std::vector<std::string> &args, const std::string &option)
{
    const auto t = tokenize(findArg(args, option), " ,");
    std::vector<float> v(t.size());
    for (size_t i = 0; i < t.size(); ++i) v[i] = strToFloat(t[i]);
    return v; // move semantics
}

/// @brief return true if the device on which it is executed uses little-endian bit ordering
inline bool isLittleEndian()
{
    unsigned int tmp = 1;
    return (*(char *)&tmp == 1);
}

/// @brief invert endianess of a type
/// @tparam T Template type to be inverted
/// @param val value to be inverted
/// @return value with reverse bytes
template <typename T>
inline T swapBytes(T val)
{
    T tmp;
    for (char *src=(char*)&val, *dst=(char*)(&tmp)+sizeof(T)-1, *end=src+sizeof(T);src!=end; *dst-- = *src++);
    return tmp;
}

/// @brief invert endianess of an array of values of a specific type
/// @tparam T Template type to be inverted
/// @param val pointer to array with values to be inverted
/// @param n number of elements in the array
template <typename T>
inline void swapBytes(T *val, int n)
{
    for (T tmp, *last = val + n; val < last; ++val) {
        for (char *src=(char*)val, *dst=(char*)(&tmp)+sizeof(T)-1, *end=src+sizeof(T); src!=end; *dst-- = *src++);
        *val = tmp;
    }
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

/// @brief Returns a string with the current local-time date stamp.
/// @details Format: "%Y-%m-%d_%H-%M-%S" (Year-Month-Day_Hour-Minute-Second).
/// @return Date-stamp string suitable for embedding in filenames.
inline std::string dateStamp() {
    const auto now = std::chrono::system_clock::now();

    // Convert it to a C-style time_t object
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm time_info;

    // Convert to local time safely (thread-safe)
#ifdef _WIN32
    localtime_s(&time_info, &now_time_t); // Windows
#else
    localtime_r(&now_time_t, &time_info); // Mac/Linux/POSIX
#endif

    // Stream it into a string using put_time
    std::ostringstream ss;
    ss << std::put_time(&time_info, "%Y-%m-%d_%H-%M-%S");

    return ss.str();
}// dateStamp

/// @brief Spinning-wheel progress indicator that overwrites a single terminal line.
/// @details Each invocation cycles through the glyphs |, /, -, \ and prints them
///          alongside a caller-supplied message, followed by a carriage return so
///          successive calls update the same line.
class Spinner {
    std::ostream& mOutStream; ///< Stream that receives the spinner output (typically std::cerr).
    const char* mBuffer;      ///< Cyclic glyph buffer "|/-\\".
    int mOffset;              ///< Index of the next glyph to display.
public:
    /// @brief Construct a Spinner writing to @a os (defaults to std::cerr).
    Spinner(std::ostream& os = std::cerr) : mOutStream(os), mBuffer{"|/-\\"}, mOffset{0} {}
    /// @brief Print one frame of the spinner with @a msg as the leading label.
    /// @param msg Message printed before the spinner glyph.
    void operator()(const std::string &msg){
        mOutStream << msg << ": " << mBuffer[mOffset] << std::setfill(' ') << std::setw(80) << "\r" << std::flush;
        mOffset = (mOffset + 1) % 4;
    }
};// Spinner

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_UTIL_HAS_BEEN_INCLUDED
