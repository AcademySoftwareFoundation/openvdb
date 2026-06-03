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
#include <cstdio> // for std::fileno (used by Spinner's TTY detection)
#include <cstring>// for std::strtok
#include <iomanip> // for std::setfill
#include <iostream>
#include <map>     // for std::multimap (used by fuzzyMatch)
#include <sstream>
#include <string>
#include <string_view>// for Spinner's operator() argument
#include <vector>
#include <sys/stat.h>// for state
#include <unistd.h>// for isatty (used by Spinner's TTY detection)
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

/// @brief Levenshtein edit distance between @a a and @a b. Counts the minimum
///        number of single-character insertions, deletions, or substitutions
///        needed to turn one string into the other. Used to rank "did you
///        mean" suggestions for typos that involve character transpositions
///        or interior insertions/deletions (cases substring matching misses).
/// @details Classic O(|a|*|b|) dynamic-programming implementation; rolls over
///          two rows so memory is O(min(|a|,|b|)). For our use case (a few
///          dozen candidate names with lengths < 32) the cost is negligible.
inline size_t levenshtein(const std::string &a, const std::string &b)
{
    if (a.empty()) return b.size();
    if (b.empty()) return a.size();
    // Ensure b is the shorter string to keep memory at O(min).
    const std::string &s = a.size() <= b.size() ? a : b;
    const std::string &t = a.size() <= b.size() ? b : a;
    const size_t m = s.size();// shorter
    const size_t n = t.size();// longer
    std::vector<size_t> prev(m + 1), curr(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = j;
    for (size_t i = 1; i <= n; ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= m; ++j) {
            const size_t cost = (t[i-1] == s[j-1]) ? 0 : 1;
            curr[j] = std::min({curr[j-1] + 1,      // insertion
                                prev[j]   + 1,      // deletion
                                prev[j-1] + cost}); // substitution
        }
        std::swap(prev, curr);
    }
    return prev[m];
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

/// @brief Turns all characters in a string into lower case. Works in-place, i.e. not copy is performed.
inline std::string &toLowerCase(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
    return str;
}

/// @brief Turns all characters in a string into upper case. Works in-place, i.e. not copy is performed.
inline std::string &toUpperCase(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::toupper(c); });
    return str;
}

/// @brief Turns all characters in a string into lower case and returns a copy.
inline std::string toLowerCase(const std::string &str)
{
    std::string tmp = str;
    return toLowerCase(tmp);
}

/// @brief Rank @a candidates by similarity to @a query and return a multimap
///        sorted so the best matches come first (lower key = better match).
///        Combines two scoring passes:
///          - Substring match in either direction (key = position of the
///            substring; 0 means an exact-prefix hit).
///          - Levenshtein edit distance (key = 1000 + distance), so substring
///            hits always rank ahead of edit-distance hits.
///        Comparisons are case-insensitive (both @a query and @a candidates
///        are lowercased internally).
/// @param query        user input to match against; empty returns no matches
/// @param candidates   list of candidate names to rank
/// @param maxDist      maximum Levenshtein distance to accept (default 2)
/// @param minCandLen   minimum candidate length for the reverse-substring and
///                     Levenshtein passes — avoids noise from 1-/2-char
///                     aliases (e.g. "-p", "-h"). Set to 0 to disable. (default 3)
/// @param useSubstring if false, skip the substring pass and rank purely by
///                     Levenshtein. Useful when the candidate domain is small
///                     and substring matches would be too noisy. (default true)
inline std::multimap<std::size_t, std::string>
fuzzyMatch(const std::string &query,
           const std::vector<std::string> &candidates,
           std::size_t maxDist      = 2,
           std::size_t minCandLen   = 3,
           bool        useSubstring = true)
{
    std::multimap<std::size_t, std::string> matches;
    if (query.empty()) return matches;
    const std::string pattern = toLowerCase(query);
    for (const std::string &cand : candidates) {
        if (cand.empty()) continue;
        const std::string candLower = toLowerCase(cand);
        if (useSubstring) {
            std::size_t key = candLower.find(pattern);
            if (key == std::string::npos && candLower.size() >= minCandLen) {
                key = pattern.find(candLower);// reverse: candidate inside the user's input
            }
            if (key != std::string::npos) {
                matches.emplace(key, cand);// substring hits rank ahead of Lev hits
                continue;
            }
        }
        if (candLower.size() < minCandLen) continue;
        const std::size_t d = levenshtein(pattern, candLower);
        if (d <= maxDist) matches.emplace(1000 + d, cand);
    }
    return matches;
}

/// @brief Turns all characters in a string into upper case and returns a copy.
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
/// @param c Set of characters to treat as whitespace (defaults to standard whitespace, new-line etc).
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
        i = stoi(s, &pos); // might throw
    } catch (const std::invalid_argument &) {
        return false;
    }
    return pos == s.size();
}

/// @brief Convert @b str into an integer.
/// @throw std::invalid_argument if @b str is not a valid integer (extra characters are not allowed).
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

/// @brief Convert @b str into a float.
/// @throw std::invalid_argument if @b str is not a valid float (extra characters are not allowed).
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

/// @brief Convert @b str into a double.
/// @throw std::invalid_argument if @b str is not a valid double (extra characters are not allowed).
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

/// @brief Convert @b str into a bool. Accepts "0"/"1" and (case-insensitive) "true"/"false".
/// @throw std::invalid_argument if @b str is not one of the accepted forms.
inline bool strToBool(const std::string &str)
{
    if (str == "1" || toLowerCase(str) == "true") return true;
    if (str == "0" || toLowerCase(str) == "false") return false;
    throw std::invalid_argument("strToBool: invalid bool \"" + str + "\"");
    return false; // "strToBool: internal error" should never happen
}

/// @brief Convert a human-readable byte-size string to an integer byte count, e.g. "2KB" -> 2048.
/// @details Recognized unit suffixes are "B", "KB", "MB", "GB", "TB", or an empty unit (bytes).
/// @throw std::invalid_argument if the numeric part fails to parse or the unit is unrecognized.
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

/// @brief Return a pseudo-random UUID version 4 string.
///
/// @details Generates a 36-character RFC 4122 v4 UUID of the form
///          "xxxxxxxx-xxxx-4xxx-Nxxx-xxxxxxxxxxxx", where the version nibble
///          is fixed at 4 and the variant nibble N ∈ {8, 9, a, b}. Two
///          mt19937_64 draws provide the 122 random bits the spec requires
///          (the remaining 6 bits are the fixed version/variant fields).
///
/// @note The PRNG is @c thread_local, so concurrent callers do not race on
///       shared state and each thread starts with an independent seed drawn
///       from std::random_device. std::random_device itself only provides a
///       32-bit seed, so a determined collision attacker could in principle
///       enumerate states — this UUID is meant for casual uniqueness (temp
///       filenames, grid IDs), not cryptographic identifiers.
inline std::string uuid()
{
    static thread_local std::mt19937_64 prng{std::random_device{}()};
    const std::uint64_t lo = prng();// time_low | time_mid | time_hi_and_version
    const std::uint64_t hi = prng();// clock_seq | node

    // Mask the version into the high nibble of time_hi_and_version (4) and
    // the variant into the high two bits of clock_seq (10xx → 8/9/a/b).
    const unsigned timeLow    = static_cast<unsigned>(lo >> 32);
    const unsigned timeMid    = static_cast<unsigned>((lo >> 16) & 0xFFFFu);
    const unsigned timeHiVer  = static_cast<unsigned>((lo & 0x0FFFu) | 0x4000u);
    const unsigned clockSeq   = static_cast<unsigned>(((hi >> 48) & 0x3FFFu) | 0x8000u);
    const unsigned nodeHi     = static_cast<unsigned>((hi >> 32) & 0xFFFFu);
    const unsigned nodeLo     = static_cast<unsigned>(hi & 0xFFFFFFFFu);

    char buf[37];// 36 hex/hyphen chars + null terminator
    std::snprintf(buf, sizeof(buf),
                  "%08x-%04x-%04x-%04x-%04x%08x",
                  timeLow, timeMid, timeHiVer, clockSeq, nodeHi, nodeLo);
    return std::string(buf, 36);
}// uuid

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
///          alongside a caller-supplied message. On a TTY the line is cleared
///          to the right via the ANSI "erase-to-EOL" escape (`\033[K`) and a
///          carriage return reuses the same line; off a TTY (pipe, log file,
///          or in-memory stream) each frame is appended on its own line so the
///          log stays readable. The destructor emits a final newline so the
///          terminal cursor lands cleanly on the next line.
class Spinner {
    std::ostream& mOutStream; ///< Stream that receives the spinner output (typically std::cerr).
    bool          mIsTty;     ///< True if @a mOutStream is connected to a terminal.
    unsigned      mOffset;    ///< Index of the next glyph to display.
    bool          mActive;    ///< Set on first call so the destructor knows whether to emit a closing newline.

    static constexpr char kGlyphs[]  = "|/-\\";
    static constexpr unsigned kCount = 4;

    /// @brief Return true if @a os routes to a terminal. Only the three
    ///        standard streams can be checked portably; any other stream
    ///        (e.g. a std::stringstream in a unit test) is treated as
    ///        non-TTY so the spinner uses the log-friendly newline form.
    static bool detectTty(const std::ostream& os) {
        if (os.rdbuf() == std::cerr.rdbuf()) return ::isatty(fileno(stderr));
        if (os.rdbuf() == std::clog.rdbuf()) return ::isatty(fileno(stderr));
        if (os.rdbuf() == std::cout.rdbuf()) return ::isatty(fileno(stdout));
        return false;
    }

public:
    /// @brief Construct a Spinner writing to @a os (defaults to std::cerr).
    Spinner(std::ostream& os = std::cerr)
        : mOutStream(os), mIsTty(detectTty(os)), mOffset(0), mActive(false) {}

    Spinner(const Spinner&) = delete;
    Spinner& operator=(const Spinner&) = delete;

    ~Spinner() {
        if (!mActive) return;
        if (mIsTty) mOutStream << "\033[K";
        mOutStream << '\n' << std::flush;
    }

    /// @brief Print one frame of the spinner with @a msg as the leading label.
    /// @param msg Message printed before the spinner glyph.
    void operator()(std::string_view msg) {
        mActive = true;
        mOutStream << msg << ": " << kGlyphs[mOffset];
        // \033[K clears from the cursor to the end of the line — replaces the
        // old fixed-width space padding so the spinner works on any terminal
        // width without wrapping. Off-TTY: newline-per-frame for readable logs.
        mOutStream << (mIsTty ? "\033[K\r" : "\n") << std::flush;
        mOffset = (mOffset + 1) % kCount;
    }

    /// @brief End the spinner line, optionally printing a final message.
    ///        Subsequent operator() calls start a fresh cycle.
    void finish(std::string_view finalMsg = {}) {
        if (mIsTty && mActive) mOutStream << "\033[K";
        mOutStream << finalMsg << '\n' << std::flush;
        mOffset = 0;
        mActive = false;
    }
};// Spinner

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_UTIL_HAS_BEEN_INCLUDED
