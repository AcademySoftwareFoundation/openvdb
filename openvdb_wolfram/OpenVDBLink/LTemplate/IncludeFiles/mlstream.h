/*
 * Copyright (c) 2019 Szabolcs Horvát.
 *
 * See the file LICENSE.txt for copying permission.
 */

#ifndef MLSTREAM_H
#define MLSTREAM_H

/** \file mlstream.h
 * \brief Auxiliary header for LTemplate to ease reading function arguments and returning values through MathLink.
 *
 * LTemplate itself does not depend on mlstream.h, so if you don't use this header,
 * feel free to remove it from your project.  mlstream.h is not meant as a general
 * MathLink interface.  It is specifically designed for handling arguments and return
 * values in conjunction with LTemplate and `LinkObject`-based functions.
 * To do this, mlstream functions are usually called in a specific sequence, as
 * illustrated below.
 *
 * Example usage:
 * \code
 * void addMult(MLINK link) {
 *   mlStream ml(link, "addMult"); // any errors messages will mention the context "addMult"
 *
 *   int i, j;
 *   ml >> mlCheckArgs(2) // read off the head List and check argument count
 *      >> i >> j;        // read two integer arguments
 *
 *   // compute the result
 *   int sum = i+j;
 *   int prod = i*j;
 *
 *   // alias for MLNewPacket, must be used before returning the result
 *   ml.newPacket();
 *
 *   // we return two results in a list
 *   ml << mlHead("List", 2)
 *      << sum << prod;
 * }
 * \endcode
 *
 * See `Documentation/Examples/LinkObject` for more examples.
 *
 * ----
 *
 * Currently, mlstream.h has direct support for sending and receiving the following types:
 *
 * **Sending**
 *
 *  - Signed integers (16-, 32- and 64-bit)
 *  - Floating point numbers
 *  - Strings (`std::string` or null-terminated C string)
 *  - `mma::RealTensorRef` and `mma::IntTensorRef` of arbitrary dimensions
 *  - `std::vector` or `std::list` holding any supported type (with optimization for `std::vector` holding numerical types)
 *  - `std::pair` holding any two supported types
 *  - Symbols (mlSymbol) or functions (mlHead)
 *
 * **Receiving**
 *
 *  - Signed integers (16-, 32- and 64-bit)
 *  - Floating point numbers
 *  - Strings (`std::string` only)
 *  - `std::vector` holding any supported type (with optimization for numerical types)
 */

#include "LTemplate.h"

#include <vector>
#include <list>
#include <utility>
#include <string>
#include <sstream>
#include <type_traits>


// Sanity checks for the sizes of MathLink integer types.
static_assert(sizeof(short) == 2,   "MathLink type size mismatch: sizeof(short) != 2.");
static_assert(sizeof(int) == 4 ,    "MathLink type size mismatch: sizeof(int) != 4.");
static_assert(sizeof(mlint64) == 8, "MathLink type size mismatch: sizeof(mlint64) != 8.");


/** \brief Wrapper for `MLINK` to allow using extractors and inserters
 *
 *  \param link is the MLINK object to wrap
 *  \param context is a string that will be prepended to any message reported using error()
 */
class mlStream {
    MLINK lp;
    std::string context;

public:
    explicit mlStream(MLINK link) : lp(link) { }
    mlStream(MLINK link, const std::string &context) : lp(link), context(context) { }

    /// Retrieve the stored `MLINK`
    MLINK link() { return lp; }

    /// Throws a \ref mma::LibraryError with a given message.
    [[ noreturn ]] void error(const std::string &err) {
        std::ostringstream msg;
        if (! context.empty())
            msg << context << ": ";
        msg << err << ".";
        throw mma::LibraryError(msg.str());
    }

    /// Equivalent to `MLNewPacket()`
    void newPacket() {
        MLNewPacket(lp);
    }
};


// Special

/// Must be the first item extracted from an mlStream, checks number of arguments and prepares for reading them.
struct mlCheckArgs {
    int argc;

    explicit mlCheckArgs(int argc) : argc(argc) { }
};

inline mlStream & operator >> (mlStream &ml, const mlCheckArgs &ca) {
    int count;

    if (! MLTestHead(ml.link(), "List", &count))
        ml.error("argument check: head \"List\" expected");

    if (count != ca.argc){
        std::ostringstream msg;
        msg << ca.argc << " argument" << (ca.argc == 1 ? "" : "s") << " expected, " << count << " received";
        ml.error(msg.str());
    }

    return ml;
}


/** \brief Used for inserting a head with the given argument count into an mlStream.
 *
 * Typically used with the head `List` when returning multiple results.
 *
 * The following example returns the complex number `3 - 2I`.
 *
 * \code
 * ml << mlHead("Complex", 2) << 3 << -2;
 * \endcode
 */
struct mlHead {
    const char *head;
    int argc;

    mlHead(const char *head, int argc) : head(head), argc(argc) { }
};

inline mlStream & operator << (mlStream &ml, const mlHead &head) {
    if (! MLPutFunction(ml.link(), head.head, head.argc)) {
        std::ostringstream msg;
        msg << "Cannot put head " << head.head << " with " << head.argc << " arguments";
        ml.error(msg.str());
    }
    return ml;
}


/** \brief Used for inserting a symbol into an mlStream
 *
 * The following example returns `True` or `False` based on a Boolean variable.
 *
 * \code
 * mlStream ml(link);
 * bool b;
 * // ...
 * ml.newPacket();
 * ml << (b ? mlSymbol("True") : mlSymbol("False"));
 * \endcode
 *
 * While this is convenient for a single result, Boolean arrays are much faster to transfer as integers.
 */
struct mlSymbol {
    const char *symbol;

    explicit mlSymbol(const char *symbol) : symbol(symbol) { }
};

inline mlStream & operator << (mlStream &ml, const mlSymbol &symbol) {
    if (! MLPutSymbol(ml.link(), symbol.symbol)) {
        std::ostringstream msg;
        msg << "Cannot put symbol " << symbol.symbol;
        ml.error(msg.str());
    }
    return ml;
}


/** \brief Used for discarding a given number of expressions from an mlStream
 *
 * The following example reads 3 arguments, but does not use the second one.
 *
 * \code
 * mlStream ml(link);
 * ml >> mlCheckArgs(3) >> x >> mlDiscard() >> y;
 * \endcode
 */
struct mlDiscard {
    const int count;
    explicit mlDiscard(int count = 1) : count(count) { }
};

inline mlStream & operator >> (mlStream &ml, const mlDiscard &drop) {
    for (int i=0; i < drop.count; ++i)
        if (! MLTransferExpression(nullptr, ml.link()))
            ml.error("Cannot discard expression");
    return ml;
}


// Basic types (integer and floating point)

#define MLSTREAM_DEF_BASIC_GET_INTEGRAL(MTYPE, CTYPE) \
    template<typename T, \
             typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) == sizeof(CTYPE), int>::type = 0 > \
    inline mlStream & operator >> (mlStream &ml, T &x) { \
        if (! MLGet ## MTYPE(ml.link(), reinterpret_cast<CTYPE *>(&x))) \
            ml.error(#MTYPE " expected"); \
        return ml; \
    }

MLSTREAM_DEF_BASIC_GET_INTEGRAL(Integer16, short)
MLSTREAM_DEF_BASIC_GET_INTEGRAL(Integer32, int)
MLSTREAM_DEF_BASIC_GET_INTEGRAL(Integer64, mlint64)


#define MLSTREAM_DEF_BASIC_GET(MTYPE, CTYPE) \
    inline mlStream & operator >> (mlStream &ml, CTYPE &x) { \
        if (! MLGet ## MTYPE(ml.link(), &x)) \
            ml.error(#MTYPE " expected"); \
        return ml; \
    }

MLSTREAM_DEF_BASIC_GET(Real32, float)
MLSTREAM_DEF_BASIC_GET(Real64, double)
MLSTREAM_DEF_BASIC_GET(Real128, mlextended_double)


#define MLSTREAM_DEF_BASIC_PUT_INTEGRAL(MTYPE, CTYPE) \
    template<typename T, \
             typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) == sizeof(CTYPE), int>::type = 0 > \
    inline mlStream & operator << (mlStream &ml, T x) { \
        if (! MLPut ## MTYPE(ml.link(), static_cast<CTYPE>(x))) \
            ml.error("Cannot return " #MTYPE); \
        return ml; \
    }

MLSTREAM_DEF_BASIC_PUT_INTEGRAL(Integer16, short)
MLSTREAM_DEF_BASIC_PUT_INTEGRAL(Integer32, int)
MLSTREAM_DEF_BASIC_PUT_INTEGRAL(Integer64, mlint64)


#define MLSTREAM_DEF_BASIC_PUT(MTYPE, CTYPE) \
    inline mlStream & operator << (mlStream &ml, CTYPE x) { \
        if (! MLPut ## MTYPE(ml.link(), x)) \
            ml.error("Cannot return " #MTYPE); \
        return ml; \
    }

MLSTREAM_DEF_BASIC_PUT(Real32, float)
MLSTREAM_DEF_BASIC_PUT(Real64, double)
MLSTREAM_DEF_BASIC_PUT(Real128, mlextended_double)


// Strings

inline mlStream & operator >> (mlStream &ml, std::string &s) {
    const unsigned char *sp;
    int bytes, chars;
    if (! MLGetUTF8String(ml.link(), &sp, &bytes, &chars))
        ml.error("String expected");
    s.assign(reinterpret_cast<const char *>(sp), bytes);
    MLReleaseUTF8String(ml.link(), sp, bytes);
    return ml;
}


inline mlStream & operator << (mlStream &ml, const std::string &s) {
    if (! MLPutUTF8String(ml.link(), reinterpret_cast<const unsigned char *>(s.c_str()), s.size()))
        ml.error("Cannot return UTF8 string");
    return ml;
}

inline mlStream & operator << (mlStream &ml, const char *s) {
    if (! MLPutString(ml.link(), s))
        ml.error("Cannot return string");
    return ml;
}


// TensorRef

inline mlStream & operator << (mlStream &ml, mma::IntTensorRef t) {
  const int maxrank  = 16;
  const int rank = t.rank();
  const mint *mdims = t.dimensions();
  int dims[maxrank];
  massert(rank <= maxrank);
  std::copy(mdims, mdims + rank, dims);
  #ifdef MINT_32
  if (! MLPutInteger32Array(ml.link(), reinterpret_cast<int *>(t.data()), dims, NULL, rank))
      ml.error("Cannot return Integer Tensor.");
  #else
  if (! MLPutInteger64Array(ml.link(), reinterpret_cast<mlint64 *>(t.data()), dims, nullptr, rank))
      ml.error("Cannot return Integer Tensor.");
  #endif
  return ml;
}

inline mlStream & operator << (mlStream &ml, mma::RealTensorRef t) {
    const int maxrank  = 16;
    const int rank = t.rank();
    const mint *mdims = t.dimensions();
    int dims[maxrank];
    massert(rank <= maxrank);
    std::copy(mdims, mdims + rank, dims);
    if (! MLPutReal64Array(ml.link(), t.data(), dims, nullptr, rank))
        ml.error("Cannot return Real Tensor");
    return ml;
}

// TODO support complex tensors


// Standard containers -- list

template<typename T>
inline mlStream & operator << (mlStream &ml, const std::list<T> &ls) {
    ml << mlHead("List", ls.size());
    for (typename std::list<T>::const_iterator i = ls.begin(); i != ls.end(); ++i)
        ml << *i;
    return ml;
}


// Standard containers -- vector

// Put signed integer element types, 16, 32 and 64 bits.
#define MLSTREAM_DEF_VEC_PUT_INTEGRAL(MTYPE, CTYPE) \
    template<typename T, \
             typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) == sizeof(CTYPE), int>::type = 0 > \
    inline mlStream & operator << (mlStream &ml, const std::vector<T> &vec) { \
        const CTYPE *data = vec.empty() ? nullptr : reinterpret_cast<const CTYPE *>(vec.data()); \
        if (! MLPut ## MTYPE ## List(ml.link(), data, vec.size())) \
            ml.error("Cannot return vector of " #MTYPE); \
        return ml; \
    }

MLSTREAM_DEF_VEC_PUT_INTEGRAL(Integer16, short)
MLSTREAM_DEF_VEC_PUT_INTEGRAL(Integer32, int)
MLSTREAM_DEF_VEC_PUT_INTEGRAL(Integer64, mlint64)

// Put floating point element types
#define MLSTREAM_DEF_VEC_PUT(MTYPE, CTYPE) \
    inline mlStream & operator << (mlStream &ml, const std::vector<CTYPE> &vec) { \
        const CTYPE *data = vec.empty() ? nullptr : vec.data(); \
        if (! MLPut ## MTYPE ## List(ml.link(), data, vec.size())) \
            ml.error("Cannot return vector of " #MTYPE); \
        return ml; \
    }

MLSTREAM_DEF_VEC_PUT(Real32, float)
MLSTREAM_DEF_VEC_PUT(Real64, double)
MLSTREAM_DEF_VEC_PUT(Real128, mlextended_double)

// Put all other types
template<typename T,
         typename std::enable_if<! (std::is_integral<T>::value && std::is_signed<T>::value && (sizeof(T) == sizeof(short) || sizeof(T) == sizeof(int) || sizeof(T) == sizeof(mlint64)) ), int>::type = 0 >
inline mlStream & operator << (mlStream &ml, const std::vector<T> &vec) {
    ml << mlHead("List", vec.size());
    for (typename std::vector<T>::const_iterator i = vec.begin(); i != vec.end(); ++i)
        ml << *i;
    return ml;
}

// Get signed integer element types
#define MLSTREAM_DEF_VEC_GET_INTEGRAL(MTYPE, CTYPE) \
    template<typename T, \
             typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) == sizeof(CTYPE), int>::type = 0> \
    inline mlStream & operator >> (mlStream &ml, std::vector<T> &vec) { \
        CTYPE *data; \
        int count; \
        if (! MLGet ## MTYPE ## List(ml.link(), &data, &count)) \
            ml.error(#MTYPE " list expected"); \
        vec.assign(data, data+count); \
        MLRelease ## MTYPE ## List(ml.link(), data, count); \
        return ml; \
    }

MLSTREAM_DEF_VEC_GET_INTEGRAL(Integer16, short)
MLSTREAM_DEF_VEC_GET_INTEGRAL(Integer32, int)
MLSTREAM_DEF_VEC_GET_INTEGRAL(Integer64, mlint64)

// Get floating point element types
#define MLSTREAM_DEF_VEC_GET(MTYPE, CTYPE) \
    inline mlStream & operator >> (mlStream &ml, std::vector<CTYPE> &vec) { \
        CTYPE *data; \
        int count; \
        if (! MLGet ## MTYPE ## List(ml.link(), &data, &count)) \
            ml.error(#MTYPE " list expected"); \
        vec.assign(data, data+count); \
        MLRelease ## MTYPE ## List(ml.link(), data, count); \
        return ml; \
    }

MLSTREAM_DEF_VEC_GET(Real32, float)
MLSTREAM_DEF_VEC_GET(Real64, double)
MLSTREAM_DEF_VEC_GET(Real128, mlextended_double)


// Get all other types
template<typename T,
         typename std::enable_if<! (std::is_integral<T>::value && std::is_signed<T>::value && (sizeof(T) == sizeof(short) || sizeof(T) == sizeof(int) || sizeof(T) == sizeof(mlint64)) ), int>::type = 0 >
inline mlStream & operator >> (mlStream &ml, std::vector<T> &vec) {
    int count;
    if (! MLTestHead(ml.link(), "List", &count))
        ml.error("Head \"List\" expected");
    vec.clear();
    vec.resize(count);
    for (auto &el : vec)
        ml >> el;
    return ml;
}


// Put an std::pair

template<typename A, typename B>
inline mlStream & operator << (mlStream &ml, const std::pair<A,B> &pair) {
    ml << mlHead("List", 2) << pair.first << pair.second;
    return ml;
}

#endif // MLSTREAM_H
