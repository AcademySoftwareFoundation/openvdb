/*
 * Copyright (c) 2019 Szabolcs Horvát.
 *
 * See the file LICENSE.txt for copying permission.
 */

/**
 * \mainpage
 *
 * This is Doxygen-generated documentation for the C++ interface of the LTemplate _Mathematica_ package.
 *
 * For the latest version of the package go to https://github.com/szhorvat/LTemplate
 *
 * See `LTemplateTutorial.nb` for an introduction and additional documentation.
 *
 * Many commented examples can be found in the `LTemplate/Documentation/Examples` directory.
 */

#ifndef LTEMPLATE_H
#define LTEMPLATE_H

/** \file
 * \brief Include this header before classes to be used with the LTemplate _Mathematica_ package.
 *
 */

#include "LTemplateCompilerSetup.h"

#ifndef LTEMPLATE_USE_CXX11
#error LTemplate requires a compiler with C++11 support.
#endif

#include "mathlink.h"
#include "WolframLibrary.h"
#include "WolframSparseLibrary.h"
#include "WolframImageLibrary.h"

#ifdef LTEMPLATE_RAWARRAY
#include "WolframRawArrayLibrary.h"
#endif

#ifdef LTEMPLATE_NUMERICARRAY
#include "WolframNumericArrayLibrary.h"
#endif

// mathlink.h defines P. It has a high potential for conflict, so we undefine it.
// It is normally only used with .tm files and it is not needed for LTemplate.
#undef P

// Sanity check for the size of mint.
#ifdef MINT_32
static_assert(sizeof(mint) == 4, "MINT_32 defined but sizeof(mint) != 4.");
#else
static_assert(sizeof(mint) == 8, "MINT_32 is not defined but sizeof(mint) != 8. Define MINT_32 when compiling on 32-bit platforms.");
#endif


#include <cstdint>
#include <complex>
#include <string>
#include <ostream>
#include <sstream>
#include <vector>
#include <map>
#include <type_traits>
#include <iterator>
#include <initializer_list>
#include <limits>

/// The namespace used by LTemplate
namespace mma {

/// Global `WolframLibraryData` object for accessing the LibraryLink API.
extern WolframLibraryData libData;

/// Complex double type for RawArrays.
typedef std::complex<double> complex_double_t;

/// Complex float type for RawArrays.
typedef std::complex<float>  complex_float_t;

/** \brief Complex number type for Tensors. Alias for \ref complex_double_t.
 *  Same as \c std::complex<double>, thus it can be used with arithmetic operators.
 */
typedef complex_double_t complex_t;

/// For use in the \ref message() function.
enum MessageType { M_INFO, M_WARNING, M_ERROR, M_ASSERT };


/** \brief Issue a _Mathematica_ message.
 *  \param msg is the text of the message
 *  \param type determines the message tag which will be used
 *
 * If `msg == nullptr`, no message will be issued. This is for compatibility with other libraries
 * that may return a null pointer instead of message text.
 *
 * \sa print()
 */
void message(const char *msg, MessageType type = M_INFO);

inline void message(const std::string &msg, MessageType type = M_INFO) { message(msg.c_str(), type); }


/** \brief Call _Mathematica_'s `Print[]`.
 *
 * \sa mout, message()
 */
inline void print(const char *msg) {
    if (libData->AbortQ())
        return; // trying to use the MathLink connection during an abort appears to break it

    MLINK link = libData->getMathLink(libData);
    MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Print", 1);
            MLPutString(link, msg);
    libData->processMathLink(link);
    int pkt = MLNextPacket(link);
    if (pkt == RETURNPKT)
        MLNewPacket(link);
}

/// Call _Mathematica_'s `Print[]`, `std::string` argument version.
inline void print(const std::string &msg) { print(msg.c_str()); }


/** \brief Can be used to output with _Mathematica_'s `Print[]` in a manner similar to `std::cout`.
 *
 * The stream _must_ be flushed to trigger printing earlier than the return of the library function.
 * This is most easily accomplished by inserting `std::endl` or `std::flush`.
 *
 * \sa print(), message()
 */
extern std::ostream mout;


/** \brief Throwing this type returns to _Mathematica_ immediately.
 *  \param msg is reported in _Mathematica_ as `LTemplate::error`.
 *  \param err is used as the LibraryFunction exit code.
 *
 *  Typical values for the exit code are the following, but any integer may be used, *except* 0 (i.e. `LIBRARY_NO_ERROR`).
 *  Using 0 in LibraryError may cause a crash, as _Mathematica_ will expect to find a return value.
 *
 *   - `LIBRARY_TYPE_ERROR`
 *   - `LIBRARY_RANK_ERROR`
 *   - `LIBRARY_DIMENSION_ERROR`
 *   - `LIBRARY_NUMERICAL_ERROR`
 *   - `LIBRARY_MEMORY_ERROR`
 *   - `LIBRARY_FUNCTION_ERROR`
 */
class LibraryError {
    const std::string msg;
    const bool has_msg;
    const int err_code;

public:
    explicit LibraryError(int err = LIBRARY_FUNCTION_ERROR) : has_msg(false), err_code(err) { }
    explicit LibraryError(const std::string &msg, int err = LIBRARY_FUNCTION_ERROR) : msg(msg), has_msg(true), err_code(err) { }

    const std::string &message() const { return msg; }
    bool has_message() const { return has_msg; }
    int error_code() const { return err_code; }

    void report() const {
        if (has_msg)
            mma::message(msg, M_ERROR);
    }
};


#ifdef NDEBUG
#define massert(condition) ((void)0)
#else
/** \brief Replacement for the standard `assert` macro. Instead of aborting the process, it throws a \ref mma::LibraryError
 *
 * As with the standard `assert` macro, define `NDEBUG` to disable assertion checks.
 * LTemplate uses massert() internally in a few places. It can be disabled this way
 * for a minor performance boost.
 */
#define massert(condition) (void)(((condition) || mma::detail::massert_impl(#condition, __FILE__, __LINE__)), 0)
#endif

namespace detail { // private
    [[ noreturn ]] inline bool massert_impl(const char *cond, const char *file, int line)
    {
        std::ostringstream msg;
        msg << cond << ", file " << file << ", line " << line;
        message(msg.str(), M_ASSERT);
        throw LibraryError();
    }
} // end namespace detail


/** \brief Return immediately to Mathematica with `LIBRARY_FUNCTION_ERROR`.
 *
 * This function is meant to be used only in situations where throwing an exception is not
 * feasible, such as from callback functions passed to C code. It should be avoided in C++
 * code as it does a `longjmp()`.
 */
[[ noreturn ]] void fatal_error();


/// Check for and honour user aborts.
inline void check_abort() {
    if (libData->AbortQ())
        throw LibraryError();
}


/// Convenience function for disowning `const char *` strings; same as `UTF8String_disown`.
inline void disownString(const char *str) {
    libData->UTF8String_disown(const_cast<char *>(str));
}


namespace detail {
    template<typename LT>
    class LTAutoFree {
        bool active;
        LT ref;
    public:
        LTAutoFree(const LT &ref) : active(true), ref(ref) { }
        LTAutoFree(LTAutoFree &&af) noexcept : LTAutoFree(af.ref) { af.active = false; }
        ~LTAutoFree() { if (active) ref.free(); }

        LTAutoFree() = delete;
        LTAutoFree(const LTAutoFree &) = delete;
        LTAutoFree & operator = (const LTAutoFree &) = delete;

        operator LT & () { return ref; }

        LT * operator -> () { return &ref; }
    };

    template<typename T>
    LTAutoFree<T> autoFree(const T &ref) { return LTAutoFree<T>(ref); }
}


/** \brief Get all instances of an LTemplate class
 *
 *  Do not use `delete` on the Class pointers in this collection or a crash may result later in the session.
 *
 */
template<typename Class>
extern const std::map<mint, Class *> &getCollection();

/** \brief Get class instance corresponding to the given managed library expression ID
 *  \tparam Class is the LTemplate class to get an instance of.
 *  \param id is the managed library expression ID.
 *
 * If no class instance corresponding to `id` exists, a \ref LibraryError will be thrown.
 *
 *  \throws LibraryError
 */
template<typename Class>
inline Class &getInstance(mint id) {
    const auto &collection = getCollection<Class>();
    auto it = collection.find(id);
    if (it == collection.end())
        throw LibraryError("Managed library expression instance does not exist.");
    return *(it->second);
}


///////////////////////////////////////  DENSE AND SPARSE ARRAY HANDLING  ///////////////////////////////////////

namespace detail { // private
    template<typename T> T * getData(MTensor t);

    template<> inline mint * getData(MTensor t) { return libData->MTensor_getIntegerData(t); }
    template<> inline double * getData(MTensor t) { return libData->MTensor_getRealData(t); }
    template<> inline complex_t * getData(MTensor t) { return reinterpret_cast< complex_t * >( libData->MTensor_getComplexData(t) ); }

    // copy data from column major format to row major format
    template<typename T, typename U>
    inline void transposedCopy(const T *from, U *to, mint nrow, mint ncol) {
        for (mint i=0; i < ncol; ++i)
            for (mint j=0; j < nrow; ++j)
                to[i + j*ncol] = from[j + i*nrow];
    }
} // end namespace detail


namespace detail { // private
    template<typename T> inline mint libraryType() {
        static_assert(std::is_same<T, T&>::value,
            "Only mint, double and mma::complex_t are allowed in mma::TensorRef<...> and mma::SparseArrayRef<...>.");
    }

    template<> inline mint libraryType<mint>()      { return MType_Integer; }
    template<> inline mint libraryType<double>()    { return MType_Real; }
    template<> inline mint libraryType<complex_t>() { return MType_Complex; }
} // end namespace detail


template<typename T> class SparseArrayRef;


/** \brief Wrapper class for `MTensor` pointers
 *  \tparam T is the type of the Tensor; must be `mint`, `double` or `mma::complex_t`.
 *
 * Specified as `LType[List, T, rank]` or `{T, rank}` in an `LTemplate` in _Mathematica_,
 * where `T` is one of `Integer`, `Real` or `Complex`.
 *
 * Note that just like `MTensor`, this class only holds a reference to a Tensor.
 * Multiple \ref TensorRef objects may refer to the same Tensor.
 *
 * \sa MatrixRef, CubeRef
 * \sa makeTensor(), makeVector(), makeMatrix(), makeCube()
 */
template<typename T>
class TensorRef {
    const MTensor t; // reminder: MTensor is a pointer type
    T * const tensor_data;
    const mint len;

    // A "null" TensorRef is used only for SparseArrayRef's ev member to handle pattern arrays
    // It cannot be publicly constructed
    TensorRef() : t(nullptr), tensor_data(nullptr), len(0) { }
    bool nullQ() const { return t == nullptr; }

    friend class SparseArrayRef<T>;

public:
    TensorRef(const MTensor &mt) :
        t(mt),
        tensor_data(detail::getData<T>(t)),
        len(libData->MTensor_getFlattenedLength(t))
    {
        detail::libraryType<T>(); // causes compile time error if T is invalid
    }

    /// The referenced \c MTensor
    MTensor tensor() const { return t; }

    /// Rank of the Tensor, same as \c MTensor_getRank
    mint rank() const { return libData->MTensor_getRank(t); }

    /// Number of elements in the Tensor, same as \c MTensor_getFlattenedLength
    mint length() const { return len; }

    /// Number of elements in the Tensor, synonym of \ref length()
    mint size() const { return length(); }

    /// Free the referenced Tensor; same as \c MTensor_free
    /**
     * Tensors created by the library with functions such as \ref makeVector() must be freed
     * after use unless they are returned to _Mathematica_.
     *
     * Warning: multiple \ref TensorRef objects may reference the same \c MTensor.
     * Freeing the \c MTensor invalidates all references to it.
     */
    void free() const { libData->MTensor_free(t); }

    /// Same as `MTensor_disown`
    void disown() const { libData->MTensor_disown(t); }

    /// Same as `MTensor_disownAll`
    void disownAll() const { libData->MTensor_disownAll(t); }

    /// Same as `MTensor_shareCount`
    mint shareCount() const { return libData->MTensor_shareCount(t); }

    /// Create a copy of the referenced Tensor
    TensorRef clone() const {
        MTensor c = nullptr;
        int err = libData->MTensor_clone(t, &c);
        if (err) throw LibraryError("MTensor_clone() failed.", err);
        return c;
    }

    const mint *dimensions() const { return libData->MTensor_getDimensions(t); }

    /// Pointer to the underlying storage of the referenced Tensor; alias of \ref begin()
    T *data() const { return tensor_data; }

    /// Index into the tensor data linearly; analogous to Flatten[tensor][[...]] in _Mathematica_
    T & operator [] (mint i) const { return tensor_data[i]; }

    /// Iterator to the beginning of the Tensor data
    T *begin() const { return data(); }

    /// Iterator past the end of the Tensor data
    T *end() const { return begin() + length(); }

    /// The type of the Tensor; may be `MType_Integer=2`, `MType_Real=3` or `MType_Complex=4`
    mint type() const { return detail::libraryType<T>(); }

    /** \brief Create a new Tensor of the given type from this Tensor's data
     *  \tparam U is the element type of the result.
     */
    template<typename U>
    TensorRef<U> convertTo() const {
        MTensor mt = nullptr;
        int err = libData->MTensor_new(detail::libraryType<U>(), rank(), dimensions(), &mt);
        if (err) throw LibraryError("MTensor_new() failed.", err);
        TensorRef<U> tr(mt);
        std::copy(begin(), end(), tr.begin());
        return tr;
    }

    /// Create a new SparseArray from the Tensor data
    SparseArrayRef<T> toSparseArray() const {
        MSparseArray sa = nullptr;
        int err = libData->sparseLibraryFunctions->MSparseArray_fromMTensor(t, NULL, &sa);
        if (err) throw LibraryError("MSparseArray_fromMTensor() failed.", err);
        return sa;
    }
};

/// @{
typedef TensorRef<mint>      IntTensorRef;
typedef TensorRef<double>    RealTensorRef;
typedef TensorRef<complex_t> ComplexTensorRef;
/// @}


/** \brief Wrapper class for `MTensor` pointers to rank-2 tensors
 *
 * Remember that \c MTensor stores data in row-major order.
 *
 * \sa TensorRef, CubeRef
 * \sa makeMatrix()
 */
template<typename T>
class MatrixRef : public TensorRef<T> {
    mint nrows, ncols;

public:
    MatrixRef(const TensorRef<T> &tr) : TensorRef<T>(tr)
    {
        if (TensorRef<T>::rank() != 2)
            throw LibraryError("MatrixRef: Matrix expected.");
        const mint *dims = TensorRef<T>::dimensions();
        nrows = dims[0];
        ncols = dims[1];
    }

    /// Number of rows in the matrix
    mint rows() const { return nrows; }

    /// Number of columns in the matrix
    mint cols() const { return ncols; }

    /// Returns 2 for a matrix
    mint rank() const { return 2; }

    /// Index into a matrix using row and column indices
    T & operator () (mint i, mint j) const { return (*this)[ncols*i + j]; }
};

/// @{
typedef MatrixRef<mint>       IntMatrixRef;
typedef MatrixRef<double>     RealMatrixRef;
typedef MatrixRef<complex_t>  ComplexMatrixRef;
/// @}


/** \brief Wrapper class for `MTensor` pointers to rank-3 tensors
 *
 * \sa TensorRef, MatrixRef
 * \sa makeCube()
 */
template<typename T>
class CubeRef : public TensorRef<T> {
    mint nslices, nrows, ncols;

public:
    CubeRef(const TensorRef<T> &tr) : TensorRef<T>(tr)
    {
        if (TensorRef<T>::rank() != 3)
            throw LibraryError("CubeRef: Rank-3 tensor expected.");
        const mint *dims = TensorRef<T>::dimensions();
        nslices = dims[0];
        nrows   = dims[1];
        ncols   = dims[2];
    }

    /// Number of rows in the cube
    mint rows() const { return nrows; }

    /// Number of columns in the cube
    mint cols() const { return ncols; }

    /// Number of slices in the cube
    mint slices() const { return nslices; }

    /// Returns 3 for a cube
    mint rank() const { return 3; }

    /// Index into a cube using slice, row, and column indices
    T & operator () (mint i, mint j, mint k) const { return (*this)[i*nrows*ncols + j*ncols + k]; }
};

/// @{
typedef CubeRef<mint>       IntCubeRef;
typedef CubeRef<double>     RealCubeRef;
typedef CubeRef<complex_t>  ComplexCubeRef;
/// @}


/** \brief Create a Tensor of the given dimensions
 *  \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`.
 *  \param dims are the dimensions
 */
template<typename T>
inline TensorRef<T> makeTensor(std::initializer_list<mint> dims) {
    MTensor t = nullptr;
    int err = libData->MTensor_new(detail::libraryType<T>(), dims.size(), dims.begin(), &t);
    if (err) throw LibraryError("MTensor_new() failed.", err);
    return t;
}

/** \brief Create a Tensor of the given dimensions.
 *  \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`.
 *  \param rank is the Tensor depth
 *  \param dims are the dimensions stored in a C array of length \c rank and type \c mint
 */
template<typename T>
inline TensorRef<T> makeTensor(mint rank, mint *dims) {
    MTensor t = nullptr;
    int err = libData->MTensor_new(detail::libraryType<T>(), rank, dims, &t);
    if (err) throw LibraryError("MTensor_new() failed.", err);
    return t;
}

/** \brief Create a Tensor of the given dimensions.
 *  \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`.
 *  \param rank is the Tensor depth
 *  \param dims are the dimensions stored in a C array of length \c rank and type \c U
 */
template<typename T, typename U>
inline TensorRef<T> makeTensor(mint rank, const U *dims) {
    std::vector<mint> d(dims, dims+rank);
    return makeTensor<T>(rank, d.data());
}


/** \brief Create a vector (rank-1 Tensor) of the given length
 * \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`
 * \param length is the vector length
 */
template<typename T>
inline TensorRef<T> makeVector(mint length) {
    return makeTensor<T>({length});
}

/** \brief Create a vector (rank-1 Tensor) of the given length and copies the contents of a C array into it
 * \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`
 * \param length is the length of the C array
 * \param data points to the contents of the C array
 */
template<typename T, typename U>
inline TensorRef<T> makeVector(mint length, const U *data) {
    TensorRef<T> t = makeVector<T>(length);
    std::copy(data, data+length, t.begin());
    return t;
}

/** \brief Create a vector (rank-1 Tensor) from an initializer list
 * \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`
 * \param values will be copied into the Tensor
 */
template<typename T>
inline TensorRef<T> makeVector(std::initializer_list<T> values) {
    TensorRef<T> t = makeVector<T>(values.size());
    std::copy(values.begin(), values.end(), t.begin());
    return t;
}


/** \brief Create a matrix (rank-2 Tensor) of the given dimensions
 * \param nrow is the number of rows
 * \param ncol is the number of columns
 * \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`
 */
template<typename T>
inline MatrixRef<T> makeMatrix(mint nrow, mint ncol) {
    return makeTensor<T>({nrow, ncol});
}

/// Create a matrix (rank-2 Tensor) of the given dimensions and copy the contents of a row-major storage C array into it
template<typename T, typename U>
inline MatrixRef<T> makeMatrix(mint nrow, mint ncol, const U *data) {
    MatrixRef<T> t = makeMatrix<T>(nrow, ncol);
    std::copy(data, data + t.size(), t.begin());
    return t;
}

/// Create a matrix (rank-2 Tensor) from a nested initializer list.
template<typename T>
inline MatrixRef<T> makeMatrix(std::initializer_list<std::initializer_list<T>> values) {
    MatrixRef<T> t = makeMatrix<T>(values.size(), values.size() ? values.begin()->size() : 0);
    T *ptr = t.data();
    for (const auto &row : values) {
        massert(row.size() == t.cols());
        for (const auto &el : row) {
            *ptr = el;
            ptr++;
        }
    }
    return t;
}

/// Create a matrix (rank-2 Tensor) of the given dimensions and copy the contents of a column-major storage C array into it
template<typename T, typename U>
inline MatrixRef<T> makeMatrixTransposed(mint nrow, mint ncol, const U *data) {
    TensorRef<T> t = makeMatrix<T>(nrow, ncol);
    detail::transposedCopy(data, t.data(), nrow, ncol);
    return t;
}


/** \brief Create a rank-3 Tensor of the given dimensions
 * \tparam T is the type of the Tensor; can be `mint`, `double` or `mma::m_complex`
 * \param nslice is the number of slices
 * \param nrow is the number of rows
 * \param ncol is the number of columns
 */
template<typename T>
inline CubeRef<T> makeCube(mint nslice, mint nrow, mint ncol) {
    return makeTensor<T>({nslice, nrow, ncol});
}

/// Create a rank-3 Tensor of the given dimensions and copy the contents of a C array into it
template<typename T, typename U>
inline CubeRef<T> makeCube(mint nslice, mint nrow, mint ncol, const U *data) {
    CubeRef<T> t = makeCube<T>(nslice, nrow, ncol);
    std::copy(data, data + t.size(), t.begin());
    return t;
}

/// Create a rank-3 Tensor from a nested initializer list
template<typename T>
inline CubeRef<T> makeCube(std::initializer_list<std::initializer_list<std::initializer_list<T>>> values) {
    size_t ns = values.size();
    size_t rs = ns ? values.begin()->size() : 0;
    size_t cs = rs ? values.begin()->begin()->size() : 0;
    CubeRef<T> t = makeCube<T>(ns, rs, cs);
    T *ptr = t.data();
    for (const auto &slice : values) {
        massert(slice.size() == rs);
        for (const auto &row : slice) {
            massert(row.size() == cs);
            for (const auto &el : row){
                *ptr = el;
                ptr++;
            }
        }
    }
    return t;
}


template<typename T> class SparseMatrixRef;

/** \brief Wrapper class for `MSparseArray` pointers
 *
 * Specified as `LType[SparseArray, T, rank]` in an `LTemplate` in _Mathematica_,
 * where `T` is one of `Integer`, `Real` or `Complex`.
 *
 * \sa SparseMatrixRef
 * \sa makeSparseArray(), makeSparseMatrix()
 */
template<typename T>
class SparseArrayRef {
    const MSparseArray sa; // reminder: MSparseArray is a pointer type
    const IntTensorRef rp; // row pointers
    const IntTensorRef ci; // column indices
    const TensorRef<T> ev; // explicit values, ev.nullQ() may be true
    T &iv;                 // implicit value

    static TensorRef<T> getExplicitValues(const MSparseArray &msa) {
        MTensor *ev = libData->sparseLibraryFunctions->MSparseArray_getExplicitValues(msa);
        if (*ev == nullptr)
            return TensorRef<T>();
        else
            return TensorRef<T>(*ev);
    }

    static IntTensorRef getColumnIndices(const MSparseArray &msa) {
        MTensor *ci = libData->sparseLibraryFunctions->MSparseArray_getColumnIndices(msa);

        // Ensure that sparse arrays always have a (possibly empty) column indices vector
        if (*ci == nullptr) {
            mint dims[2] = {0, libData->sparseLibraryFunctions->MSparseArray_getRank(msa)};
            libData->MTensor_new(MType_Integer, 2, dims, ci);
        }

        return *ci;
    }

    static T &getImplicitValue(const MSparseArray &msa) {
        MTensor *mt = libData->sparseLibraryFunctions->MSparseArray_getImplicitValue(msa);
        return *(detail::getData<T>(*mt));
    }

    friend class SparseMatrixRef<T>;

public:
    SparseArrayRef(const MSparseArray &msa) :
        sa(msa),
        rp(*(libData->sparseLibraryFunctions->MSparseArray_getRowPointers(msa))),
        ci(getColumnIndices((msa))),
        ev(getExplicitValues(msa)),
        iv(getImplicitValue(msa))
    {
        detail::libraryType<T>(); // causes compile time error if T is invalid
    }

    /// The references `MSparseArray`
    MSparseArray sparseArray() const { return sa; }

    /// Rank of the SparseArray
    mint rank() const { return libData->sparseLibraryFunctions->MSparseArray_getRank(sa); }

    /// Dimensions of the SparseArray
    const mint *dimensions() const { return libData->sparseLibraryFunctions->MSparseArray_getDimensions(sa); }

    /// The number of explicitly stored positions
    mint length() const { return ci.length(); /* use ci because ev may be null */}

    /// The number of explicitly stored positions, alias for length()
    mint size() const { return length(); }

    void free() const { libData->sparseLibraryFunctions->MSparseArray_free(sa); }
    void disown() const { libData->sparseLibraryFunctions->MSparseArray_disown(sa); }
    void disownAll() const { libData->sparseLibraryFunctions->MSparseArray_disownAll(sa); }

    mint shareCount() const { return libData->sparseLibraryFunctions->MSparseArray_shareCount(sa); }

    /// Create a copy of the referenced SparseArray
    SparseArrayRef clone() const {
        MSparseArray c = nullptr;
        int err = libData->sparseLibraryFunctions->MSparseArray_clone(sa, &c);
        if (err) throw LibraryError("MSparseArray_clone() failed.", err);
        return c;
    }

    /** \brief  Create a new integer Tensor containing the indices of non-default (i.e. explicit) values in the sparse array.
     *
     *  The positions use 1-based indexing.
     *
     *  You are responsible for freeing this data structure using the TensorRef::free() function when done using it.
     */
    IntTensorRef explicitPositions() const {
        MTensor mt = nullptr;
        int err = libData->sparseLibraryFunctions->MSparseArray_getExplicitPositions(sa, &mt);
        if (err) throw LibraryError("MSParseArray_getExplicitPositions() failed.", err);

        // Workaround for MSparseArray_getExplicitPositions() returning a non-empty rank-0 MTensor
        // when the SparseArray has no explicit positions: in this case we manually construct
        // a rank-2 0-by-n empty integer MTensor and return that instead.
        if (libData->MTensor_getRank(mt) == 0) {
            libData->MTensor_free(mt);
            return makeMatrix<mint>(0, rank());
        }
        else {
            return IntTensorRef(mt);
        }
    }

    /** \brief The column indices of the SparseArray's internal CSR representation, as an integer Tensor.
     *
     * This function is useful when converting a SparseArray for use with another library that also
     * uses a CSR or CSC representation.
     *
     * The result is either a rank-2 Tensor or an empty one. The indices are 1-based.
     *
     * The result `MTensor` is part of the `MSparseArray` data structure and will be destroyed at the same time with it.
     * Clone it before returning it to the kernel using \ref clone().
     */
    IntTensorRef columnIndices() const {
        return ci;
    }

    /** \brief The row pointers of the SparseArray's internal CSR representation, as a rank-1 integer Tensor.
     *
     * This function is useful when converting a SparseArray for use with another library that also
     * uses a CSR or CSC representation.
     *
     * The result `MTensor` is part of the `MSparseArray` data structure and will be destroyed at the same time with it.
     * Clone it before returning it to the kernel using \ref clone().
     */
    IntTensorRef rowPointers() const { return rp; }

    /// Does the SparseArray store explicit values?  Pattern arrays do not have explicit values.
    bool explicitValuesQ() const { return ! ev.nullQ(); }

    /** \brief The explicit values in the SparseArray as a Tensor.
     *
     * The result `MTensor` is part of the `MSparseArray` data structure and will be destroyed at the same time with it.
     * Clone it before returning it to the kernel using \ref clone().
     *
     * For pattern arrays, which do not have explicit values, a \ref LibraryError exception is thrown.
     *
     * \sa explicitValuesQ()
     */
    TensorRef<T> explicitValues() const {
        if (ev.nullQ())
            throw LibraryError("SparseArrayRef::explicitValues() called on pattern array.");
        return ev;
    }

    /// The implicit value (also call background or default value) of the SparseArray
    T &implicitValue() const { return iv; }

    /** \brief Creates a new SparseArray in which explicitly stored values that are equal to the current implicit value are eliminated.
     *
     * Useful when the explicit values or the implicit value has been changed, and a recomputation of the CSR structure is desired.
     *
     * Should not be used on a pattern array.
     */
    SparseArrayRef resetImplicitValue() const {
        MSparseArray msa = nullptr;
        int err = libData->sparseLibraryFunctions->MSparseArray_resetImplicitValue(sa, NULL, &msa);
        if (err) throw LibraryError("MSparseArray_resetImplicitValue() failed.", err);
        return msa;
    }

    /** \brief Creates a new SparseArray based on a new implicit value
     *  \param iv is the new implicit value
     */
    SparseArrayRef resetImplicitValue(const T &iv) const {
        MSparseArray msa = nullptr;

        MTensor it = nullptr;
        int err = libData->MTensor_new(detail::libraryType<T>(), 0, nullptr, &it);
        if (err) throw LibraryError("MTensor_new() failed.", err);
        *detail::getData<T>(it) = iv;

        err = libData->sparseLibraryFunctions->MSparseArray_resetImplicitValue(sa, it, &msa);
        libData->MTensor_free(it);
        if (err) throw LibraryError("MSparseArray_resetImplicitValue() failed.", err);

        return msa;
    }

    /// Creates a new Tensor (dense array) containing the same elements as the SparseArray
    TensorRef<T> toTensor() const {
        MTensor t = nullptr;
        int err = libData->sparseLibraryFunctions->MSparseArray_toMTensor(sa, &t);
        if (err) throw LibraryError("MSparseArray_toMTensor() failed.", err);
        return t;
    }

    /// The element type of the SparseArray; may be `MType_Integer=2`, `MType_Real=3` or `MType_Complex=4`
    mint type() const { return detail::libraryType<T>(); }
};


/** \brief Wrapper class for rank-2 SparseArrays
 *
 * \sa SparseArrayRef
 * \sa makeSparseMatrix()
 */
template<typename T>
class SparseMatrixRef : public SparseArrayRef<T> {
    mint ncols, nrows;

    using SparseArrayRef<T>::rp;
    using SparseArrayRef<T>::ci;
    using SparseArrayRef<T>::ev;
    using SparseArrayRef<T>::iv;

public:
    using SparseArrayRef<T>::dimensions;
    using SparseArrayRef<T>::size;
    using SparseArrayRef<T>::explicitValuesQ;

    /// Bidirectional iterator for enumerating the explicitly stored values and positions of a sparse matrix.
    class iterator : public std::iterator<std::bidirectional_iterator_tag, T> {
        const SparseMatrixRef *smp;
        mint row_index, index;

        friend class SparseMatrixRef;

        iterator(const SparseMatrixRef *smp, const mint &row_index, const mint &index) :
            smp(smp),
            row_index(row_index),
            index(index)
        { /* empty */ }

    public:

        iterator() = default;
        iterator(const iterator &) = default;
        iterator &operator = (const iterator &) = default;

        /** \brief Access explicit value.
         *
         * Should not be used with pattern arrays. There is no safety check for this.
         */
        T &operator *() const { return smp->ev[index]; }

        bool operator == (const iterator &it) const { return index == it.index; }
        bool operator != (const iterator &it) const { return index != it.index; }

        iterator &operator ++ () {
            index++;
            while (smp->rp[row_index+1] == index && index < smp->size())
                row_index++;
            return *this;
        }

        iterator operator ++ (int) {
            iterator it = *this;
            operator++();
            return it;
        }

        iterator &operator -- () {
            while (smp->rp[row_index] == index && row_index > 0)
                row_index--;
            index--;
            return *this;
        }

        iterator operator -- (int) {
            iterator it = *this;
            operator--();
            return it;
        }

        mint row() const { return row_index; } ///< Row of the referenced element (0-based indexing)
        mint col() const { return smp->ci[index]-1; } ///< Column of the referenced element (0-based indexing)
    };

    SparseMatrixRef(const SparseArrayRef<T> &sa) : SparseArrayRef<T>(sa)
    {
        if (SparseArrayRef<T>::rank() != 2)
            throw LibraryError("SparseMatrixRef: Matrix expected.");
        const mint *dims = dimensions();
        nrows = dims[0];
        ncols = dims[1];
    }

    /// Number of rows in the sparse matrix
    mint rows() const { return nrows; }

    /// Number of columns in the sparse matrix
    mint cols() const { return ncols; }

    mint rank() const { return 2; }

    /** \brief Index into a sparse matrix (read-only, 0-based)
     *
     * This operator provides read access only.
     * For write access to explicit values, use explicitValues().
     */
    T operator () (mint i, mint j) const {
        if (! explicitValuesQ())
            throw LibraryError("SparseMatrixRef: cannot index into a pattern array.");

        // if (i,j) is explicitly stored, it must be located between
        // the following array indices in ev and ci:
        mint lower = rp[i];
        mint upper = rp[i+1];

        // look for the index j between those locations:
        mint *cp = std::lower_bound(&ci[lower], &ci[upper], j+1);
        if (cp == &ci[upper]) // no bound found
            return iv;
        else if (*cp == j+1)  // found a bound equal to the sought column index
            return ev[lower + (cp - &ci[lower])];
        else                  // column index not found
            return iv;
    }

    /** \brief Iterator to beginning of explicit values and positions
     *
     *  If you only need explicit values, not explicit positions, use `sparseArray.explicitValues().begin()` instead.
     *
     * \sa SparseMatrixRef::iterator
     */
    iterator begin() const {
        mint row_index = 0;
        while (rp[row_index+1] == 0 && row_index < rp.size())
            row_index++;
        return iterator{this, row_index, 0};
    }

    /// Iterator to the end of explicit values and positions
    iterator end() const {
        return iterator{this, rows(), size()};
    }
};


/** \brief Create a new SparseArray from a set of positions and values.
 *
 * \param pos is the list of explicitly stored positions using 1-based indexing.
 * \param vals is the list of explicitly stored values.
 * \param dims is a list of the sparse array dimensions.
 * \param imp is the implicit value.
 */
template<typename T>
inline SparseArrayRef<T> makeSparseArray(IntMatrixRef pos, TensorRef<T> vals, IntTensorRef dims, T imp = 0) {
    int err;

    massert(pos.cols() == dims.size());
    massert(pos.rows() == vals.size());

    MTensor it = nullptr;
    err = libData->MTensor_new(detail::libraryType<T>(), 0, nullptr, &it);
    if (err)
        throw LibraryError("makeSparseArray: MTensor_new() failed.", err);
    *detail::getData<T>(it) = imp;

    MSparseArray sa = nullptr;
    err = libData->sparseLibraryFunctions->MSparseArray_fromExplicitPositions(pos.tensor(), vals.tensor(), dims.tensor(), it, &sa);
    libData->MTensor_free(it);
    if (err)
        throw LibraryError("makeSparseArray: MSparseArray_fromExplicitPositions() failed.", err);

    // MSparseArray_fromExplicitPositions() will return a pattern array when the positions array is empty.
    // When this happens, we manually insert an explicit values array to ensure that this function
    // never returns a pattern array.
    MTensor *ev;
    ev = libData->sparseLibraryFunctions->MSparseArray_getExplicitValues(sa);
    if (*ev == nullptr) {
        mint evdims[1] = {0};
        libData->MTensor_new(detail::libraryType<T>(), 1, evdims, ev);
    }

    return sa;
}

/** \brief Create a new sparse matrix from a set of positions and values.
 *
 * \param pos is the list of explicitly stored positions using 1-based indexing.
 * \param vals is the list of explicitly stored values.
 * \param nrow is the number of matrix rows.
 * \param ncol is the number of matrix columns.
 * \param imp is the implicit value.
 */
template<typename T>
inline SparseMatrixRef<T> makeSparseMatrix(IntMatrixRef pos, TensorRef<T> vals, mint nrow, mint ncol, T imp = 0) {
    massert(pos.cols() == 2);

    auto dims = detail::autoFree(makeVector<mint>({nrow, ncol}));
    SparseMatrixRef<T> sa = makeSparseArray(pos, vals, dims, imp);

    return sa;
}


//////////////////////////////////////////  RAW ARRAY HANDLING  //////////////////////////////////////////

#ifdef LTEMPLATE_RAWARRAY

namespace detail { // private
    template<typename T> inline rawarray_t libraryRawType() {
        static_assert(std::is_same<T, T&>::value,
            "Only int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, complex_float_t, complex_double_t are allowed in mma::RawArrayRef<...>.");
    }

    template<> inline rawarray_t libraryRawType<int8_t>()   { return MRawArray_Type_Bit8;   }
    template<> inline rawarray_t libraryRawType<uint8_t>()  { return MRawArray_Type_Ubit8;  }
    template<> inline rawarray_t libraryRawType<int16_t>()  { return MRawArray_Type_Bit16;  }
    template<> inline rawarray_t libraryRawType<uint16_t>() { return MRawArray_Type_Ubit16; }
    template<> inline rawarray_t libraryRawType<int32_t>()  { return MRawArray_Type_Bit32;  }
    template<> inline rawarray_t libraryRawType<uint32_t>() { return MRawArray_Type_Ubit32; }
    template<> inline rawarray_t libraryRawType<int64_t>()  { return MRawArray_Type_Bit64;  }
    template<> inline rawarray_t libraryRawType<uint64_t>() { return MRawArray_Type_Ubit64; }
    template<> inline rawarray_t libraryRawType<float>()    { return MRawArray_Type_Real32; }
    template<> inline rawarray_t libraryRawType<double>()   { return MRawArray_Type_Real64; }
    template<> inline rawarray_t libraryRawType<complex_float_t>()  { return MRawArray_Type_Float_Complex; }
    template<> inline rawarray_t libraryRawType<complex_double_t>() { return MRawArray_Type_Double_Complex; }

    inline const char *rawTypeMathematicaName(rawarray_t rt) {
        switch (rt) {
        case MRawArray_Type_Ubit8:          return "UnsignedInteger8";
        case MRawArray_Type_Bit8:           return "Integer8";
        case MRawArray_Type_Ubit16:         return "UnsignedInteger16";
        case MRawArray_Type_Bit16:          return "Integer16";
        case MRawArray_Type_Ubit32:         return "UnsignedInteger32";
        case MRawArray_Type_Bit32:          return "Integer32";
        case MRawArray_Type_Ubit64:         return "UnsignedInteger64";
        case MRawArray_Type_Bit64:          return "Integer64";
        case MRawArray_Type_Real32:         return "Real32";
        case MRawArray_Type_Real64:         return "Real64";
        case MRawArray_Type_Float_Complex:  return "Complex32";
        case MRawArray_Type_Double_Complex: return "Complex64";
        case MRawArray_Type_Undef:          return "Undefined";
        default:                            return "Unknown"; // should never reach here
        }
    }

} // end namespace detail


template<typename T> class RawArrayRef;


/// Wrapper class for `MRawArray` pointers; unspecialized base class. Typically used through \ref RawArrayRef.
class GenericRawArrayRef {
    const MRawArray ra;
    const mint len;

public:
    GenericRawArrayRef(const MRawArray &mra) :
        ra(mra),
        len(libData->rawarrayLibraryFunctions->MRawArray_getFlattenedLength(mra))
    { }

    /// Returns the referenced \c MRawArray
    MRawArray rawArray() const { return ra; }

    /// Returns the rank of the RawArray, same as \c MRawArray_getRank
    mint rank() const { return libData->rawarrayLibraryFunctions->MRawArray_getRank(ra); }

    /// Returns the number of elements in the RawArray, same as \c MRawArray_getFlattenedLength
    mint length() const { return len; }

    /// Returns the number of elements in the RawArray, synonym of \ref length()
    mint size() const { return length(); }

    /// Frees the referenced RawArray, same as \c MRawArray_free
    /**
     * Warning: multiple \ref RawArrayRef objects may reference the same \c MRawArray.
     * Freeing the \c MRawArray invalidates all references to it.
     */
    void free() const { libData->rawarrayLibraryFunctions->MRawArray_free(ra); }

    void disown() const { libData->rawarrayLibraryFunctions->MRawArray_disown(ra); }
    void disownAll() const { libData->rawarrayLibraryFunctions->MRawArray_disownAll(ra); }

    mint shareCount() const { return libData->rawarrayLibraryFunctions->MRawArray_shareCount(ra); }

    const mint *dimensions() const { return libData->rawarrayLibraryFunctions->MRawArray_getDimensions(ra); }

    /// Creates a copy of the referenced RawArray
    GenericRawArrayRef clone() const {
        MRawArray c = nullptr;
        int err = libData->rawarrayLibraryFunctions->MRawArray_clone(rawArray(), &c);
        if (err) throw LibraryError("MRawArray_clone() failed.", err);
        return c;
    }

    /** \brief Convert to the given type of RawArray; same as `MRawArray_convertType`
     *  \tparam U is the element type of the result
     */
    template<typename U>
    RawArrayRef<U> convertTo() const {
        MRawArray res = libData->rawarrayLibraryFunctions->MRawArray_convertType(ra, detail::libraryRawType<U>());
        if (! res)
            throw LibraryError("MRawArray_convertType() failed.");
        return res;
    }

    rawarray_t type() const { return libData->rawarrayLibraryFunctions->MRawArray_getType(ra); }
};


/// Wrapper class for `MRawArray` pointers. Available only in _Mathematica_ 10.4 and later. With _Mathematica_ 12.0 or later, use \ref NumericArrayRef instead.
template<typename T>
class RawArrayRef : public GenericRawArrayRef {
    T * const array_data;

    void checkType() {
        rawarray_t received = GenericRawArrayRef::type();
        rawarray_t expected = detail::libraryRawType<T>();
        if (received != expected) {
            std::ostringstream err;
            err << "RawArray of type " << detail::rawTypeMathematicaName(received) << " received, "
                << detail::rawTypeMathematicaName(expected) << " expected.";
            throw LibraryError(err.str(), LIBRARY_TYPE_ERROR);
        }
    }

public:

    RawArrayRef(const MRawArray &mra) :
        GenericRawArrayRef(mra),
        array_data(reinterpret_cast<T *>(libData->rawarrayLibraryFunctions->MRawArray_getData(mra)))
    {
        checkType();
    }

    // explicit conversion required to prevent accidental auto-conversion between RawArrays of different types
    explicit RawArrayRef(const GenericRawArrayRef &gra) :
        GenericRawArrayRef(gra),
        array_data(reinterpret_cast<T *>(libData->rawarrayLibraryFunctions->MRawArray_getData(gra.rawArray())))
    {
        checkType();
    }

    /// Creates a copy of the referenced RawArray
    RawArrayRef clone() const {
        MRawArray c = nullptr;
        int err = libData->rawarrayLibraryFunctions->MRawArray_clone(rawArray(), &c);
        if (err) throw LibraryError("MRawArray_clone() failed.", err);
        return c;
    }

    /// Returns a pointer to the underlying storage of the corresponding \c MRawArray
    T *data() const { return array_data; }

    T & operator [] (mint i) const { return array_data[i]; }

    T *begin() const { return data(); }
    T *end() const { return begin() + length(); }

    rawarray_t type() const { return detail::libraryRawType<T>(); }
};

/** \brief Creates a RawArray of the given dimensions
 *  \tparam T is the array element type
 *  \param dims are the array dimensions
 */
template<typename T>
inline RawArrayRef<T> makeRawArray(std::initializer_list<mint> dims) {
    MRawArray ra = nullptr;
    int err = libData->rawarrayLibraryFunctions->MRawArray_new(detail::libraryRawType<T>(), dims.size(), dims.begin(), &ra);
    if (err) throw LibraryError("MRawArray_new() failed.", err);
    return ra;
}

/** \brief Create a RawArray of the given dimensions.
 *  \tparam T is the array element type
 *  \param rank is the RawArray depth
 *  \param dims are the dimensions stored in a C array of length \c rank and type \c mint
 */
template<typename T>
inline RawArrayRef<T> makeRawArray(mint rank, const mint *dims) {
    MRawArray ra = nullptr;
    int err = libData->rawarrayLibraryFunctions->MRawArray_new(detail::libraryRawType<T>(), rank, dims, &ra);
    if (err) throw LibraryError("MRawArray_new() failed.", err);
    return ra;
}

/** \brief Create a RawArray of the given dimensions.
 *  \tparam T is the array element type
 *  \param rank is the RawArray depth
 *  \param dims are the dimensions stored in a C array of length \c rank and type \c U
 */
template<typename T, typename U>
inline RawArrayRef<T> makeRawArray(mint rank, const U *dims) {
    std::vector<mint> d(dims, dims+rank);
    return makeRawArray<T>(rank, d.data());
}

/** \brief Creates a rank-1 RawArray of the given length
 *  \tparam T is the array element type
 *  \param length is the vector length
 */
template<typename T>
inline RawArrayRef<T> makeRawVector(mint length) {
    return makeRawArray<T>({length});
}

/** \brief Creates a rank-1 RawArray of the given type from a C array of the corresponding type
 *  \tparam T is the array element type
 *  \param length is the vector length
 *  \param data will be copied into the raw vector
 */
template<typename T>
inline RawArrayRef<T> makeRawVector(mint length, const T *data) {
    auto ra = makeRawVector<T>(length);
    std::copy(data, data+length, ra.begin());
    return ra;
}

#endif // LTEMPLATE_RAWARRAY


//////////////////////////////////////////  NUMERIC ARRAY HANDLING  //////////////////////////////////////////

/*
 * NumericArray was added in _Mathematica_ 12.0. It is identical to the earlier RawArray, and converts seamlessly from it,
 * but it is now fully documented. The old RawArray LibraryLink interface is still present, so we keep RawArrayRef for
 * backwards compatibility.
 *
 * Differences in the RawArray and NumericArray LibraryLink API:
 *  - Some enum values in MNumericArray_Data_Type have been renamed.
 *  - The convertType function has changed significantly:
 *      - Now returns an error code.
 *      - Can write into an existing NumericArray of the same dimensions. First argument should point to NULL
 *        (not be NULL) if a new NumericArray is to be created.
 *      - Conversion method can be specified. It corresponds to the third argument of the NumericArray _Mathematica_ function.
 *      - Tolerance can be given, to be used with floating point conversions.
 */

#ifdef LTEMPLATE_NUMERICARRAY

/** \brief Decimal digits per one bit (binary digit).
 *
 * Equal to \f$ \ln 2 / \ln 10 \f$.
 * This constant is useful in setting the `tolerance` option of \ref GenericNumericArrayRef::convertTo.
 */
constexpr double decimalDigitsPerBit = 0.3010299956639812;

namespace detail { // private
    template<typename T> inline numericarray_data_t libraryNumericType() {
        static_assert(std::is_same<T, T&>::value,
            "Only int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, complex_float_t, complex_double_t are allowed in mma::NumericArrayRef<...>.");
    }

    template<> inline numericarray_data_t libraryNumericType<int8_t>()   { return MNumericArray_Type_Bit8;   }
    template<> inline numericarray_data_t libraryNumericType<uint8_t>()  { return MNumericArray_Type_UBit8;  }
    template<> inline numericarray_data_t libraryNumericType<int16_t>()  { return MNumericArray_Type_Bit16;  }
    template<> inline numericarray_data_t libraryNumericType<uint16_t>() { return MNumericArray_Type_UBit16; }
    template<> inline numericarray_data_t libraryNumericType<int32_t>()  { return MNumericArray_Type_Bit32;  }
    template<> inline numericarray_data_t libraryNumericType<uint32_t>() { return MNumericArray_Type_UBit32; }
    template<> inline numericarray_data_t libraryNumericType<int64_t>()  { return MNumericArray_Type_Bit64;  }
    template<> inline numericarray_data_t libraryNumericType<uint64_t>() { return MNumericArray_Type_UBit64; }
    template<> inline numericarray_data_t libraryNumericType<float>()    { return MNumericArray_Type_Real32; }
    template<> inline numericarray_data_t libraryNumericType<double>()   { return MNumericArray_Type_Real64; }
    template<> inline numericarray_data_t libraryNumericType<complex_float_t>()  { return MNumericArray_Type_Complex_Real32; }
    template<> inline numericarray_data_t libraryNumericType<complex_double_t>() { return MNumericArray_Type_Complex_Real64; }

    inline const char *numericTypeMathematicaName(numericarray_data_t rt) {
        switch (rt) {
        case MNumericArray_Type_UBit8:          return "UnsignedInteger8";
        case MNumericArray_Type_Bit8:           return "Integer8";
        case MNumericArray_Type_UBit16:         return "UnsignedInteger16";
        case MNumericArray_Type_Bit16:          return "Integer16";
        case MNumericArray_Type_UBit32:         return "UnsignedInteger32";
        case MNumericArray_Type_Bit32:          return "Integer32";
        case MNumericArray_Type_UBit64:         return "UnsignedInteger64";
        case MNumericArray_Type_Bit64:          return "Integer64";
        case MNumericArray_Type_Real32:         return "Real32";
        case MNumericArray_Type_Real64:         return "Real64";
        case MNumericArray_Type_Complex_Real32:  return "Complex32";
        case MNumericArray_Type_Complex_Real64: return "Complex64";
        case MNumericArray_Type_Undef:          return "Undefined";
        default:                            return "Unknown"; // should never reach here
        }
    }

} // end namespace detail


template<typename T> class NumericArrayRef;


/// Wrapper class for `MNumericArray` pointers; unspecialized base class. Typically used through \ref NumericArrayRef.
class GenericNumericArrayRef {
    const MNumericArray na;
    const mint len;

public:

    GenericNumericArrayRef(const MNumericArray &mra) :
        na(mra),
        len(libData->numericarrayLibraryFunctions->MNumericArray_getFlattenedLength(mra))
    { }

    /// Returns the referenced \c MNumericArray
    MNumericArray numericArray() const { return na; }

    /// Returns the rank of the NumericArray, same as \c MNumericArray_getRank
    mint rank() const { return libData->numericarrayLibraryFunctions->MNumericArray_getRank(na); }

    /// Returns the number of elements in the NumericArray, same as \c MNumericArray_getFlattenedLength
    mint length() const { return len; }

    /// Returns the number of elements in the NumericArray, synonym of \ref length()
    mint size() const { return length(); }

    /// Frees the referenced NumericArray, same as \c MNumericArray_free
    /**
     * Warning: multiple \ref NumericArrayRef objects may reference the same \c MNumericArray.
     * Freeing the \c MNumericArray invalidates all references to it.
     */
    void free() const { libData->numericarrayLibraryFunctions->MNumericArray_free(na); }

    void disown() const { libData->numericarrayLibraryFunctions->MNumericArray_disown(na); }
    void disownAll() const { libData->numericarrayLibraryFunctions->MNumericArray_disownAll(na); }

    mint shareCount() const { return libData->numericarrayLibraryFunctions->MNumericArray_shareCount(na); }

    const mint *dimensions() const { return libData->numericarrayLibraryFunctions->MNumericArray_getDimensions(na); }

    /// Creates a copy of the referenced NumericArray
    GenericNumericArrayRef clone() const {
        MNumericArray c = nullptr;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_clone(numericArray(), &c);
        if (err) throw LibraryError("MNumericArray_clone() failed.", err);
        return c;
    }

    /// Used in \ref convertTo to specify the element type conversion method. The names are consistent with the coercion methods of `NumericArray` in Mathematica.
    enum ConversionMethod {
        Check = 1, ///< Throw a \ref LibraryError if any values do not fit in target type without modification.
        ClipAndCheck, ///< Clip to the target range and check that the values fit in the target type.
        Coerce, ///< Coerce values into the target type.
        ClipAndCoerce, ///< Clip to the target range and coerce into the target type.
        Round, ///< Round reals to integers.
        ClipAndRound, ///< Clip to the range and round reals to integers.
        Scale, ///< Scale to the range (undocumented as of _Mathematica_ 12.0).
        ClipAndScale ///< Clip and scale to the range (undocumented as of _Mathematica_ 12.0).
    };

    /** \brief Convert to the given type of NumericArray; same as `MNumericArray_convertType`
     *  \tparam U is the element type of the result
     *
     * \param method is the conversion method (see \ref ConversionMethod)
     * \param tolerance is the tolerance in decimal digits for checking whether a value can be accurately represented using the target type
     *
     * The default tolerance corresponds to one binary digit for consistency with \c NumericArray in Mathematica.
     * Use \ref decimalDigitsPerBit for conveniently specifying the tolerance in bits instead of decimal digits.
     * Use e.g. \c 3*decimalDigitsPerBit to set 3 bits of tolerance.
     *
     * If any of the element values cannot be converted to the target type with the specified conversion method, a \ref LibraryError
     * will be thrown.
     */
    template<typename U>
    NumericArrayRef<U> convertTo(ConversionMethod method = ClipAndRound, double tolerance = decimalDigitsPerBit) const {
        MNumericArray res = nullptr;
        auto err = libData->numericarrayLibraryFunctions->MNumericArray_convertType(&res, na, detail::libraryNumericType<U>(), static_cast<numericarray_convert_method_t>(method), tolerance);
        if (err)
            throw LibraryError("MNumericArray_convertType() failed. Check that all values can be converted to the target type using the specified method and tolerance.");
        return res;
    }

    template<typename U>
    NumericArrayRef<U> convertTo(numericarray_convert_method_t method, double tolerance = decimalDigitsPerBit) const {
        return convertTo<U>(ConversionMethod(method), tolerance);
    }

    numericarray_data_t type() const { return libData->numericarrayLibraryFunctions->MNumericArray_getType(na); }
};


/// Wrapper class for `MNumericArray` pointers. Available only in _Mathematica_ 12.0 and later.
template<typename T>
class NumericArrayRef : public GenericNumericArrayRef {
    T * const array_data;

    void checkType() {
        numericarray_data_t received = GenericNumericArrayRef::type();
        numericarray_data_t expected = detail::libraryNumericType<T>();
        if (received != expected) {
            std::ostringstream err;
            err << "NumericArray of type " << detail::numericTypeMathematicaName(received) << " received, "
                << detail::numericTypeMathematicaName(expected) << " expected.";
            throw LibraryError(err.str(), LIBRARY_TYPE_ERROR);
        }
    }

public:

    NumericArrayRef(const MNumericArray &mra) :
        GenericNumericArrayRef(mra),
        array_data(reinterpret_cast<T *>(libData->numericarrayLibraryFunctions->MNumericArray_getData(mra)))
    {
        checkType();
    }

    // explicit conversion required to prevent accidental auto-conversion between NumericArrays of different types
    explicit NumericArrayRef(const GenericNumericArrayRef &gra) :
        GenericNumericArrayRef(gra),
        array_data(reinterpret_cast<T *>(libData->numericarrayLibraryFunctions->MNumericArray_getData(gra.numericArray())))
    {
        checkType();
    }

    /// Creates a copy of the referenced NumericArray
    NumericArrayRef clone() const {
        MNumericArray c = nullptr;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_clone(numericArray(), &c);
        if (err) throw LibraryError("MNumericArray_clone() failed.", err);
        return c;
    }

    /// Returns a pointer to the underlying storage of the corresponding \c MNumericArray
    T *data() const { return array_data; }

    T & operator [] (mint i) const { return array_data[i]; }

    T *begin() const { return data(); }
    T *end() const { return begin() + length(); }

    numericarray_data_t type() const { return detail::libraryNumericType<T>(); }
};

/** \brief Creates a NumericArray of the given dimensions
 *  \tparam T is the array element type
 *  \param dims are the array dimensions
 */
template<typename T>
inline NumericArrayRef<T> makeNumericArray(std::initializer_list<mint> dims) {
    MNumericArray na = nullptr;
    int err = libData->numericarrayLibraryFunctions->MNumericArray_new(detail::libraryNumericType<T>(), dims.size(), dims.begin(), &na);
    if (err) throw LibraryError("MNumericArray_new() failed.", err);
    return na;
}

/** \brief Create a NumericArray of the given dimensions.
 *  \tparam T is the array element type
 *  \param rank is the NumericArray depth
 *  \param dims are the dimensions stored in a C array of length \c rank and type \c mint
 */
template<typename T>
inline NumericArrayRef<T> makeNumericArray(mint rank, const mint *dims) {
    MNumericArray na = nullptr;
    int err = libData->numericarrayLibraryFunctions->MNumericArray_new(detail::libraryNumericType<T>(), rank, dims, &na);
    if (err) throw LibraryError("MNumericArray_new() failed.", err);
    return na;
}

/** \brief Create a NumericArray of the given dimensions.
 *  \tparam T is the array element type
 *  \param rank is the NumericArray depth
 *  \param dims are the dimensions stored in a C array of length \c rank and type \c U
 */
template<typename T, typename U>
inline NumericArrayRef<T> makeNumericArray(mint rank, const U *dims) {
    std::vector<mint> d(dims, dims+rank);
    return makeNumericArray<T>(rank, d.data());
}

/** \brief Creates a rank-1 NumericArray of the given length
 *  \tparam T is the array element type
 *  \param length is the vector length
 */
template<typename T>
inline NumericArrayRef<T> makeNumericVector(mint length) {
    return makeNumericArray<T>({length});
}

/** \brief Creates a rank-1 NumericArray of the given type from a C array of the corresponding type
 *  \tparam T is the array element type
 *  \param length is the vector length
 *  \param data will be copied into the numeric vector
 */
template<typename T>
inline NumericArrayRef<T> makeNumericVector(mint length, const T *data) {
    auto na = makeNumericVector<T>(length);
    std::copy(data, data+length, na.begin());
    return na;
}

#endif // LTEMPLATE_NUMERICARRAY



//////////////////////////////////////////  IMAGE HANDLING  //////////////////////////////////////////


/* While the C++ standard does not guarantee that sizeof(bool) == 1, this is currently
 * the case for most implementations. This was verified at https://gcc.godbolt.org/
 * across multiple platforms and architectures in October 2017.
 *
 * The ABIs used by major operating systems also specify a bool or C99 _Bool of size 1
 * https://github.com/rust-lang/rfcs/pull/954#issuecomment-169820630
 *
 * Thus it seems safe to require sizeof(bool) == 1. A safety check is below.
 */
static_assert(sizeof(bool) == 1, "The bool type is expected to be of size 1.");

/* We use a new set of types for image elements. These all correspond to raw_t_... types
 * from WolframImageLibrary.h with the exception of im_bit_t, which is bool.
 * This is so that it will be distinct from im_byte_t.
 */
/// @{
typedef bool            im_bit_t;    ///< `"Bit"`, `MImage_Type_Bit`
typedef unsigned char   im_byte_t;   ///< `"Byte"`, `MImage_Type_Bit8`
typedef unsigned short  im_bit16_t;  ///< `"Bit16"`, `MImage_Type_Bit16`
typedef float           im_real32_t; ///< `"Real32"`, `MImage_Type_Real32`
typedef double          im_real_t;   ///< `"Real"`, `MImage_Type_Real`
typedef im_byte_t       im_bit8_t;   ///< Alias for \ref im_byte_t
/// @}

/** \brief Returns the value representing "white" for the give pixel type
 *  \tparam T is the pixel type
 *
 * For integer types, this is the highest value that can be stored.
 *
 * For floating point types, it is 1.0. Higher values can be stored, but will be
 * clipped during display or conversion.
 *
 * The value representing "black" is always 0.
 */
template<typename T>
inline T imageMax() { return std::numeric_limits<T>::max(); }

template<>
inline im_bit_t imageMax() { return 1; }

template<>
inline im_real32_t imageMax() { return 1.0f; }

template<>
inline im_real_t imageMax() { return 1.0; }

namespace detail { // private
    template<typename T> inline imagedata_t libraryImageType() {
        static_assert(std::is_same<T, T&>::value,
            "Only im_bit_t, im_byte_t, im_bit16_t, im_real32_t, im_real_t are allowed in mma::ImageRef<...>.");
    }

    template<> inline imagedata_t libraryImageType<im_bit_t>()    { return MImage_Type_Bit;   }
    template<> inline imagedata_t libraryImageType<im_byte_t>()   { return MImage_Type_Bit8;  }
    template<> inline imagedata_t libraryImageType<im_bit16_t>()  { return MImage_Type_Bit16;  }
    template<> inline imagedata_t libraryImageType<im_real32_t>() { return MImage_Type_Real32; }
    template<> inline imagedata_t libraryImageType<im_real_t>()   { return MImage_Type_Real;  }


    inline const char *imageTypeMathematicaName(imagedata_t it) {
        switch (it) {
        case MImage_Type_Bit:       return "Bit";
        case MImage_Type_Bit8:      return "Byte";
        case MImage_Type_Bit16:     return "Bit16";
        case MImage_Type_Real32:    return "Real32";
        case MImage_Type_Real:      return "Real";
        case MImage_Type_Undef:     return "Undefined";
        default:                    return "Unknown"; // should never reach here
        }
    }

} // end namespace detail


template<typename T> class ImageRef;
template<typename T> class Image3DRef;

/** \brief Wrapper class for `MImage` pointers referring to 2D images; unspecialized base class. Typically used through \ref ImageRef.
 *
 * \sa GenericImage3DRef
 */
class GenericImageRef {
    const MImage im;
    const mint len;
    const mint nrows, ncols, nchannels;
    const bool interleaved, alphaChannel;

public:
    GenericImageRef(const MImage &mim) :
        im(mim),
        len(libData->imageLibraryFunctions->MImage_getFlattenedLength(im)),
        nrows(libData->imageLibraryFunctions->MImage_getRowCount(im)),
        ncols(libData->imageLibraryFunctions->MImage_getColumnCount(im)),
        nchannels(libData->imageLibraryFunctions->MImage_getChannels(im)),
        interleaved(libData->imageLibraryFunctions->MImage_interleavedQ(im)),
        alphaChannel(libData->imageLibraryFunctions->MImage_alphaChannelQ(im))
    {
        massert(libData->imageLibraryFunctions->MImage_getRank(im) == 2);
    }

    /// The referenced \c MImage
    MImage image() const { return im; }

    /// Total number of pixels in all image channels. Same as \c MImage_getFlattenedLength. \sa channelSize()
    mint length() const { return len; }

    /// Total number of pixels in all image channels; synonym of \ref length(). \sa channelSize()
    mint size() const { return length(); }

    /// The number of image rows
    mint rows() const { return nrows; }

    /// The number of image columns
    mint cols() const { return ncols; }

    /// The number of pixels in a single image channel
    mint channelSize() const { return rows()*cols(); }

    /// Returns 2 for 2D images
    mint rank() const { return 2; }

    /// The number of image channels
    mint channels() const { return nchannels; }

    /// The number of non-alpha channels. Same as \ref channels() if the image has no alpha channel; one less otherwise.
    mint nonAlphaChannels() const { return alphaChannelQ() ? channels()-1 : channels(); }

    /// Are channels stored in interleaved mode? Interleaved means e.g. `rgbrgbrgb...` instead of `rrr...ggg...bbb...` for an RGB image.
    bool interleavedQ() const { return interleaved; }

    /// Does the image have an alpha channel?
    bool alphaChannelQ() const { return alphaChannel; }

    /// The image colour space
    colorspace_t colorSpace() const {
        return libData->imageLibraryFunctions->MImage_getColorSpace(im);
    }

    /// Create a copy of the referenced Image
    GenericImageRef clone() const {
        MImage c = nullptr;
        int err = libData->imageLibraryFunctions->MImage_clone(image(), &c);
        if (err) throw LibraryError("MImage_clone() failed.", err);
        return c;
    }

    /// Free the referenced \c MImage; same as \c MImage_free
    void free() const { libData->imageLibraryFunctions->MImage_free(im); }

    void disown() const { libData->imageLibraryFunctions->MImage_disown(im); }
    void disownAll() const { libData->imageLibraryFunctions->MImage_disownAll(im); }

    mint shareCount() const { return libData->imageLibraryFunctions->MImage_shareCount(im); }

    /** \brief Convert to the given type of Image; same as `MImage_convertType`.
     *  \param interleaving specifies whether to store the data in interleaved mode. See \ref interleavedQ()
     *  \tparam U is the pixel type of the result.
     *
     *  Returns a new image that must be freed explicitly unless returned to the kernel. See \ref free().
     */
    template<typename U>
    ImageRef<U> convertTo(bool interleaving) const {
        MImage res = libData->imageLibraryFunctions->MImage_convertType(im, detail::libraryImageType<U>(), interleaving);
        if (! res)
            throw LibraryError("MImage_convertType() failed.");
        return res;
    }

    /// Convert to the given type of Image. The interleaving mode is preserved.
    template<typename U>
    ImageRef<U> convertTo() const { return convertTo<U>(interleavedQ()); }

    /// Returns the image/pixel type
    imagedata_t type() const {
        return libData->imageLibraryFunctions->MImage_getDataType(im);
    }
};


/** \brief Wrapper class for `MImage` pointers referring to 3D images; unspecialized base class. Typically used through \ref Image3DRef.
 *
 * \sa GenericImageRef
 */
class GenericImage3DRef {
    const MImage im;
    const mint len;
    const mint nrows, ncols, nslices, nchannels;
    const bool interleaved, alphaChannel;

public:
    GenericImage3DRef(const MImage &mim) :
        im(mim),
        len(libData->imageLibraryFunctions->MImage_getFlattenedLength(im)),
        nrows(libData->imageLibraryFunctions->MImage_getRowCount(im)),
        ncols(libData->imageLibraryFunctions->MImage_getColumnCount(im)),
        nslices(libData->imageLibraryFunctions->MImage_getSliceCount(im)),
        nchannels(libData->imageLibraryFunctions->MImage_getChannels(im)),
        interleaved(libData->imageLibraryFunctions->MImage_interleavedQ(im)),
        alphaChannel(libData->imageLibraryFunctions->MImage_alphaChannelQ(im))
    {
        massert(libData->imageLibraryFunctions->MImage_getRank(im) == 3);
    }

    /// The referenced \c MImage
    MImage image() const { return im; }

    /// The total number of pixels in all image channels. Same as \c MImage_getFlattenedLength. \sa channelSize()
    mint length() const { return len; }

    /// The total number of pixels in all image channels, synonym of \ref length(). \sa channelSize()
    mint size() const { return length(); }

    /// The number of image rows
    mint rows() const { return nrows; }

    /// The number of image columns
    mint cols() const { return ncols; }

    /// The number of image slices in the Image3D
    mint slices() const { return nslices; }

    /// The number of pixels in a single image channel.
    mint channelSize() const { return slices()*rows()*cols(); }

    /// Returns 3 for 3D images.
    mint rank() const { return 3; }

    /// The number of image channels
    mint channels() const { return nchannels; }

    /// The number of non-alpha channels. Same as \ref channels() if the image has no alpha channel; one less otherwise.
    mint nonAlphaChannels() const { return alphaChannelQ() ? channels()-1 : channels(); }

    /// Are channels stored in interleaved mode? Interleaved means e.g. `rgbrgbrgb...` instead of `rrr...ggg...bbb...` for an RGB image.
    bool interleavedQ() const { return interleaved; }

    /// Does the image have an alpha channel?
    bool alphaChannelQ() const { return alphaChannel; }

    /// The image colour space
    colorspace_t colorSpace() const {
        return libData->imageLibraryFunctions->MImage_getColorSpace(im);
    }

    /// Create a copy of the referenced Image3D
    GenericImage3DRef clone() const {
        MImage c = nullptr;
        int err = libData->imageLibraryFunctions->MImage_clone(image(), &c);
        if (err) throw LibraryError("MImage_clone() failed.", err);
        return c;
    }

    /// Free the referenced \c MImage; same as \c MImage_free
    void free() const { libData->imageLibraryFunctions->MImage_free(im); }

    void disown() const { libData->imageLibraryFunctions->MImage_disown(im); }
    void disownAll() const { libData->imageLibraryFunctions->MImage_disownAll(im); }

    mint shareCount() const { return libData->imageLibraryFunctions->MImage_shareCount(im); }

    /** \brief Convert to the given type of Image3D; same as `MImage_convertType`.
     *  \param interleaving specifies whether to store the data in interleaved mode. See \ref interleavedQ()
     *  \tparam U is the pixel type of the result.
     *
     *  Returns a new image that must be freed explicitly unless returned to the kernel. See \ref free().
     */
    template<typename U>
    Image3DRef<U> convertTo(bool interleaving) const {
        MImage res = libData->imageLibraryFunctions->MImage_convertType(im, detail::libraryImageType<U>(), interleaving);
        if (! res)
            throw LibraryError("MImage_convertType() failed.");
        return res;
    }

    /// Convert to the given type of Image3D. The interleaving mode is preserved.
    template<typename U>
    Image3DRef<U> convertTo() const { return convertTo<U>(interleavedQ()); }

    /// Returns the image/pixel type
    imagedata_t type() const {
        return libData->imageLibraryFunctions->MImage_getDataType(im);
    }
};


template<typename T>
class pixel_iterator : public std::iterator<std::random_access_iterator_tag, T> {
    T *ptr;
    ptrdiff_t step;

    friend ImageRef<T>;
    friend Image3DRef<T>;

    pixel_iterator(T *ptr, ptrdiff_t step) :
        ptr(ptr), step(step)
    { }

public:

    pixel_iterator() = default;
    pixel_iterator(const pixel_iterator &) = default;
    pixel_iterator &operator = (const pixel_iterator &) = default;


    bool operator == (const pixel_iterator &it) const { return ptr == it.ptr; }
    bool operator != (const pixel_iterator &it) const { return ptr != it.ptr; }

    T &operator *() const { return *ptr; }

    pixel_iterator &operator ++ () {
        ptr += step;
        return *this;
    }

    pixel_iterator operator ++ (int) {
        pixel_iterator it = *this;
        operator++();
        return it;
    }

    pixel_iterator &operator -- () {
        ptr -= step;
        return *this;
    }

    pixel_iterator operator -- (int) {
        pixel_iterator it = *this;
        operator--();
        return it;
    }

    pixel_iterator operator + (ptrdiff_t n) const { return pixel_iterator(ptr + n*step, step); }

    pixel_iterator operator - (ptrdiff_t n) const { return pixel_iterator(ptr - n*step, step); }

    ptrdiff_t operator - (const pixel_iterator &it) const { return (ptr - it.ptr)/step; }

    T & operator [] (mint i) { return ptr[i*step]; }

    bool operator < (const pixel_iterator &it) const { return ptr < it.ptr; }
    bool operator > (const pixel_iterator &it) const { return ptr > it.ptr; }
    bool operator <= (const pixel_iterator &it) const { return ptr <= it.ptr; }
    bool operator >= (const pixel_iterator &it) const { return ptr >= it.ptr; }
};


/** \brief Wrapper class for `MImage` pointers referring to 2D images.
 *  \tparam T is the pixel type of the image. It corresponds to _Mathematica_'s `ImageType` as per the table below:
 *
 * `ImageType` | C++ type      |  Alias for
 * ------------|---------------|-------------------
 *  `"Bit"`    | `im_bit_t`    |  `bool`
 *  `"Byte"`   | `im_byte_t`   |  `unsigned char`
 *  `"Bit16"`  | `im_bit16_t`  |  `unsigned short`
 *  `"Real32"` | `im_real32_t` |  `float`
 *  `"Real"`   | `im_real_t`   |  `double`
 *
 * Note that this class only holds a reference to an Image. Multiple \ref ImageRef
 * or \ref GenericImageRef objects may point to the same Image.
 *
 * \sa Image3DRef
 */
template<typename T>
class ImageRef : public GenericImageRef {
    T * const image_data;

public:

    /// Random access iterator for accessing the pixels of a single image channel in order
    typedef class pixel_iterator<T> pixel_iterator;

    ImageRef(const MImage &mim) :
        GenericImageRef(mim),
        image_data(reinterpret_cast<T *>(libData->imageLibraryFunctions->MImage_getRawData(mim)))
    { }

    // explicit conversion required to prevent accidental auto-conversion between Images of different types or Image and Image3D
    /// Cast a GenericImageRef to a type-specialized 2D image. The pixel type must agree with that of the generic image, otherwise an error is thrown.
    explicit ImageRef(const GenericImageRef &gim) :
        GenericImageRef(gim),
        image_data(reinterpret_cast<T *>(libData->imageLibraryFunctions->MImage_getRawData(gim.image())))
    {
        imagedata_t received = gim.type();
        imagedata_t expected = detail::libraryImageType<T>();
        if (received != expected) {
            std::ostringstream err;
            err << "Image of type " << detail::imageTypeMathematicaName(received) << " received, "
                << detail::imageTypeMathematicaName(expected) << " expected.";
            throw LibraryError(err.str(), LIBRARY_TYPE_ERROR);
        }
    }

    /// Create a copy of the referenced Image
    ImageRef clone() const {
        MImage c = nullptr;
        int err = libData->imageLibraryFunctions->MImage_clone(image(), &c);
        if (err) throw LibraryError("MImage_clone() failed.", err);
        return c;
    }

    /// Pointer to the image data; synonym of \ref begin()
    T *data() const { return image_data; }

    /// Iterator to the beginning of the image data
    T *begin() const { return data(); }

    /// Iterator past the end of the image data
    T *end() const { return begin() + length(); }

    /// Pixel iterator to the beginning of \p channel
    pixel_iterator pixelBegin(mint channel) const {
        if (interleavedQ())
            return pixel_iterator(image_data + channel, channels());
        else
            return pixel_iterator(image_data + channelSize()*channel, 1);
    }

    /// Pixel iterator past the end of \p channel
    pixel_iterator pixelEnd(mint channel) const {
        return pixelBegin(channel) + channelSize();
    }

    /// Index into the Image by pixel coordinates and channel
    T &operator ()(mint row, mint col, mint channel = 0) const {
        if (interleavedQ())
            return image_data[row*cols()*channels() + col*channels()+ channel];
        else
            return image_data[channel*rows()*cols() + row*cols() + col];
    }

    /// The element / pixel type of the Image
    imagedata_t type() const { return detail::libraryImageType<T>(); }
};


/** \brief Wrapper class for `MImage` pointers referring to 3D images.
 *  \tparam T is the element type of the image. It corresponds to _Mathematica_'s `ImageType` as per the table below:
 *
 * `ImageType` | C++ type      |  Alias for
 * ------------|---------------|-------------------
 *  `"Bit"`    | `im_bit_t`    |  `bool`
 *  `"Byte"`   | `im_byte_t`   |  `unsigned char`
 *  `"Bit16"`  | `im_bit16_t`  |  `unsigned short`
 *  `"Real32"` | `im_real32_t` |  `float`
 *  `"Real"`   | `im_real_t`   |  `double`
 *
 * Note that this class only holds a reference to an Image3D. Multiple \ref Image3DRef
 * or \ref GenericImage3DRef objects may point to the same Image3D.
 *
 * \sa ImageRef
 */
template<typename T>
class Image3DRef : public GenericImage3DRef {
    T * const image_data;

public:

    /// Random access iterator for accessing the pixels of a single image channel in order
    typedef class pixel_iterator<T> pixel_iterator;

    Image3DRef(const MImage &mim) :
        GenericImage3DRef(mim),
        image_data(reinterpret_cast<T *>(libData->imageLibraryFunctions->MImage_getRawData(mim)))
    { }

    // explicit conversion required to prevent accidental auto-conversion between Image3Ds of different types or Image and Image3D
    /// Cast a GenericImage3DRef to a type-specialized 3D image. The pixel type must agree with that of the generic image, otherwise an error is thrown.
    explicit Image3DRef(const GenericImage3DRef &gim) :
        GenericImage3DRef(gim),
        image_data(reinterpret_cast<T *>(libData->imageLibraryFunctions->MImage_getRawData(gim.image())))
    {
        imagedata_t received = gim.type();
        imagedata_t expected = detail::libraryImageType<T>();
        if (received != expected) {
            std::ostringstream err;
            err << "Image3D of type " << detail::imageTypeMathematicaName(received) << " received, "
                << detail::imageTypeMathematicaName(expected) << " expected.";
            throw LibraryError(err.str(), LIBRARY_TYPE_ERROR);
        }
    }

    /// Create a copy of the referenced Image3D
    Image3DRef clone() const {
        MImage c = nullptr;
        int err = libData->imageLibraryFunctions->MImage_clone(image(), &c);
        if (err) throw LibraryError("MImage_clone() failed.", err);
        return c;
    }

    /// Pointer to the image data
    T *data() const { return image_data; }

    /// Iterator to the beginning of the image data
    T *begin() const { return data(); }

    /// Iterator past the end of the image data
    T *end() const { return begin() + length(); }

    /// Pixel iterator to the beginning of \p channel
    pixel_iterator pixelBegin(mint channel) const {
        if (interleavedQ())
            return pixel_iterator(image_data + channel, channels());
        else
            return pixel_iterator(image_data + channelSize()*channel, 1);
    }

    /// Pixel iterator past the end of \p channel
    pixel_iterator pixelEnd(mint channel) const {
        return pixelBegin(channel) + channelSize();
    }

    /// Index into the Image3D by pixel coordinates and channel
    T &operator ()(mint slice, mint row, mint col, mint channel = 0) const {
        if (interleavedQ())
            return image_data[slice*rows()*cols()*channels() + row*cols()*channels() + col*channels() + channel];
        else
            return image_data[channel*slices()*rows()*cols() + slice*rows()*cols() + row*cols() + col];
    }

    /// The element / pixel type of the Image3D
    imagedata_t type() const { return detail::libraryImageType<T>(); }
};


/** \brief Create a new Image
 *  \tparam T is the pixel type
 *  \param width of the image (number of columns)
 *  \param height of the image (number of rows)
 *  \param channels is the number of image channels
 *  \param interleaving specifies whether to store the data in interleaved mode
 *  \param colorspace may be one of `MImage_CS_Automatic`, `MImage_CS_Gray`, `MImage_CS_RGB`, `MImage_CS_HSB`, `MImage_CS_CMYK`, `MImage_CS_XYZ`, `MImage_CS_LUV`, `MImage_CS_LAB`, `MImage_CS_LCH`
 */
template<typename T>
inline ImageRef<T> makeImage(mint width, mint height, mint channels = 1, bool interleaving = true, colorspace_t colorspace = MImage_CS_Automatic) {
    MImage mim = nullptr;
    libData->imageLibraryFunctions->MImage_new2D(width, height, channels, detail::libraryImageType<T>(), colorspace, interleaving, &mim);
    return mim;
}

/** \brief Create a new Image3D
 *  \tparam T is the pixel type
 *  \param slices is the number of image slices
 *  \param width of individual slices (number of rows)
 *  \param height of individual slices (number of columns)
 *  \param channels is the number of image channels
 *  \param interleaving specifies whether to store the data in interleaved mode
 *  \param colorspace may be one of `MImage_CS_Automatic`, `MImage_CS_Gray`, `MImage_CS_RGB`, `MImage_CS_HSB`, `MImage_CS_CMYK`, `MImage_CS_XYZ`, `MImage_CS_LUV`, `MImage_CS_LAB`, `MImage_CS_LCH`
 *
 */
template<typename T>
inline Image3DRef<T> makeImage3D(mint slices, mint width, mint height, mint channels = 1, bool interleaving = true, colorspace_t colorspace = MImage_CS_Automatic) {
    MImage mim = nullptr;
    libData->imageLibraryFunctions->MImage_new3D(slices, width, height, channels, detail::libraryImageType<T>(), colorspace, interleaving, &mim);
    return mim;
}

} // end namespace mma

#endif // LTEMPLATE_H

