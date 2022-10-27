// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBCOMMON_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBCOMMON_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetUtil.h>

#include <openvdb/tree/LeafManager.h>

#include <ctime>
#include <vector>

#include <functional>// for std::hash
#include <algorithm> // for std::min(), std::max()
#include <array> // for std::array
#include <limits>

#include "LTemplate.h"

using namespace openvdb;
using namespace openvdb::tools;
using namespace openvdb::math;


namespace openvdbmma {
namespace types {

template<class T>
struct scalar_type : std::integral_constant<bool, std::is_floating_point<T>::value> {};

template<class T>
struct integer_type : std::integral_constant<bool, std::is_integral<T>::value> {};

template<class T>
struct vector_type : std::integral_constant<bool, VecTraits<T>::IsVec> {};

template<class T>
struct bool_type : std::integral_constant<bool, std::is_same<bool, T>::value> {};

template<class T>
struct mask_type : std::integral_constant<bool, std::is_same<openvdb::ValueMask, T>::value> {};

// anything that can work with Image, Image3D, functionality
template<class T>
struct pixel_type : std::integral_constant<bool,
        std::is_floating_point<T>::value ||
        std::is_same<uint8_t, T>::value ||
        std::is_same<bool, T>::value ||
        std::is_same<openvdb::ValueMask, T>::value
    > {};

template<typename V>
inline void
scalar_type_assert()
{
    static_assert(scalar_type<V>::value, "This method requires a scalar grid.");
}

template<typename V>
inline void
int_type_assert()
{
    static_assert(integer_type<V>::value, "This method requires an integer grid.");
}

template<typename V>
inline void
vector_type_assert()
{
    static_assert(vector_type<V>::value, "This method requires a vector grid.");
}

template<typename V>
inline void
non_bool_type_assert()
{
    static_assert(!bool_type<V>::value, "This method requires a non-bool grid.");
}

template<typename V>
inline void
non_mask_type_assert()
{
    static_assert(!mask_type<V>::value, "This method requires a non-mask grid.");
}

template<typename V>
inline void
pixel_type_assert()
{
    static_assert(pixel_type<V>::value, "This method requires a pixel grid.");
}

} // namespace types

namespace utils {

void
initialize()
{
    if (!Grid<UInt32Tree>::isRegistered()) {
        Grid<tree::Tree4<uint8_t, 5, 4, 3>::Type>::registerGrid();
        Grid<UInt32Tree>::registerGrid();
        Grid<Vec2DTree>::registerGrid();
        Grid<Vec2ITree>::registerGrid();
        Grid<Vec2STree>::registerGrid();
    }

    openvdb::initialize();
}

//////////////// Metadata utilities

#define META_CREATED           "creation_date"
#define META_CUTOFF_DISTANCE   "cutoff_distance"
#define META_DESCRIPTION       "description"
#define META_GAMMA_ADJUSTMENT  "gamma_adjustment"
#define META_LAST_MODIFIED     "last_modified_date"
#define META_SCALING_FACTOR    "scaling_factor"


/////////////// global enumerations

// grid class
enum {
    GC_UNKNOWN = 0,
    GC_LEVELSET,
    GC_FOGVOLUME
};

// filtering methods
enum {
    MEAN_FILTER = 0,
    MEDIAN_FILTER,
    GAUSSIAN_FILTER,
    LAPLACIAN_FILTER,
    MEAN_CURVATURE_FILTER
};

// resampling methods
enum {
    RS_NEAREST = 0,
    RS_LINEAR,
    RS_QUADRATIC
};


/////////////// Value transforms

template<typename TreeType, typename XForm>
struct TransformAllValues
{
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    explicit TransformAllValues(const XForm& f) : op(f) {}

    TransformAllValues(const TransformAllValues& other, tbb::split) : op(other.op) {}

    void operator()(RootT& node) const
    {
        for (auto iter = node.beginValueAll(); iter; ++iter)
            iter.setValueOnly(op(*iter));
    }

    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        for (auto iter = node.beginValueAll(); iter; ++iter)
            iter.setValueOnly(op(*iter));
    }

    void operator()(LeafT& leaf) const
    {
        for (auto iter = leaf.beginValueAll(); iter; ++iter)
            iter.setValueOnly(op(*iter));
    }

    void join(const TransformAllValues& other) {}

private:

    XForm op;
};

template<typename TreeType, typename XForm>
struct TransformActiveLeafValues
{
    TransformActiveLeafValues(const XForm& f) : op(f) {}

    template <typename LeafNodeType>
    void operator()(LeafNodeType& leaf, size_t) const
    {
        for (auto iter = leaf.beginValueOn(); iter; ++iter)
            iter.setValue(op(*iter));
    }

private:

    XForm op;
};

template<typename TreeType, typename XForm>
void transformAllValues(TreeType& tree, XForm xop)
{
    TransformAllValues<TreeType, XForm> op(xop);

    tree::NodeManager<TreeType> node(tree);
    node.reduceTopDown(op);
}

template<typename TreeType, typename XForm>
void transformActiveLeafValues(TreeType& tree, XForm xop)
{
    TransformActiveLeafValues<TreeType, XForm> op(xop);

    tree::LeafManager<TreeType> leafNodes(tree);
    leafNodes.foreach(op);
}

} // namespace utils
} // namespace openvdbmma


/////////////// Interface between WL and OpenVDB interrupter

namespace mma {
namespace interrupt {

struct LLInterrupter
{
    LLInterrupter () { wlLibData = mma::libData; }

    LLInterrupter (WolframLibraryData libData) : wlLibData(libData) {}

    void start(const char* name = nullptr) { (void)name; }

    void end() {}

    inline bool wasInterrupted(int percent = -1) { (void)percent; return wlLibData->AbortQ(); }

private:

    WolframLibraryData wlLibData;
};

template <typename T>
inline bool wasInterrupted(T* i, int percent = -1) { return i && i->wasInterrupted(percent); }

} // namespace interrupt
} // namespace mma


/////////////// Tensor utilities

namespace mma {

template<typename T>
class VectorRef : public TensorRef<T>
{
    mint nelems;

public:
    VectorRef(const TensorRef<T> &tr) : TensorRef<T>(tr)
    {
        if (TensorRef<T>::rank() != 1)
            throw LibraryError("VectorRef: rank 1 tensor expected.");
        const mint *dims = TensorRef<T>::dimensions();
        nelems = dims[0];
    }

    mint size() const { return nelems; }

    mint rank() const { return 1; }

    T & operator () (mint i) const { return (*this)[i]; }
};

typedef VectorRef<mint>       IntVectorRef;
typedef VectorRef<double>     RealVectorRef;


template<typename T>
class TesseractRef : public TensorRef<T> {
    mint nhslices, nslices, nrows, ncols;

public:
    TesseractRef(const TensorRef<T> &tr) : TensorRef<T>(tr)
    {
        if (TensorRef<T>::rank() != 4)
            throw LibraryError("TesseractRef: Rank-4 tensor expected.");
        const mint *dims = TensorRef<T>::dimensions();
        nhslices   = dims[0];
        nslices = dims[1];
        nrows   = dims[2];
        ncols   = dims[3];
    }

    /// Number of rows in the tesseract
    mint rows() const { return nrows; }

    /// Number of columns in the tesseract
    mint cols() const { return ncols; }

    /// Number of slices in the tesseract
    mint slices() const { return nslices; }

    /// Number of hyper slices in the tesseract
    mint hyper_slices() const { return nhslices; }

    /// Returns 3 for a cube
    mint rank() const { return 4; }

    /// Index into a cube using slice, row, and column indices
    T & operator () (mint i, mint j, mint k, mint l) const { return (*this)[i*nslices*nrows*ncols + j*nrows*ncols + k*ncols + l]; }
};

typedef TesseractRef<mint>       IntTesseractRef;
typedef TesseractRef<double>     RealTesseractRef;

template<typename T>
inline TesseractRef<T> makeTesseract(mint nhslice, mint nslice, mint nrow, mint ncol) {
    return makeTensor<T>({nhslice, nslice, nrow, ncol});
}


class RGBRef : public TensorRef<double>
{
public:
    RGBRef(const TensorRef<double> &tr) : TensorRef<double>(tr)
    {
        if (TensorRef<double>::rank() != 1)
            throw LibraryError("RGBRef: rank 1 tensor expected.");
        if (TensorRef<double>::size() != 3)
            throw LibraryError("RGBRef: length 3 vector expected.");

        (*this)[0] = (*this)[0] < 0.0 ? 0.0 : (*this)[0] > 1.0 ? 1.0 : (*this)[0];
        (*this)[1] = (*this)[1] < 0.0 ? 0.0 : (*this)[1] > 1.0 ? 1.0 : (*this)[1];
        (*this)[2] = (*this)[2] < 0.0 ? 0.0 : (*this)[2] > 1.0 ? 1.0 : (*this)[2];
    }

    inline double r() const { return (*this)[0]; }
    inline double g() const { return (*this)[1]; }
    inline double b() const { return (*this)[2]; }
};


template<typename T>
class CoordinatesRef : public TensorRef<T>
{
    mint ncoords;

public:
    CoordinatesRef(const TensorRef<T> &tr) : TensorRef<T>(tr)
    {
        if (TensorRef<T>::rank() != 2)
            throw LibraryError("CoordinatesRef: Matrix expected.");
        const mint *dims = TensorRef<T>::dimensions();
        if (dims[1] != 3)
            throw LibraryError("CoordinatesRef: 3D coordinates expected.");
        ncoords = dims[0];
    }

    mint size() const { return ncoords; }

    T x(mint i) const { return (*this)[3*i]; }
    T y(mint i) const { return (*this)[3*i+1]; }
    T z(mint i) const { return (*this)[3*i+2]; }
};

typedef CoordinatesRef<mint>    IntCoordinatesRef;
typedef CoordinatesRef<double>  RealCoordinatesRef;

template<typename T>
inline CoordinatesRef<T> makeCoordinatesList(mint n) { return makeMatrix<T>(n, 3); }

template<typename T>
class Bounds3DRef : public TensorRef<T>
{
public:
    Bounds3DRef(const TensorRef<T> &tr) : TensorRef<T>(tr)
    {
        if (TensorRef<T>::rank() != 2)
            throw LibraryError("Bounds3DRef: 3D bounding box expected.");
        const mint *dims = TensorRef<T>::dimensions();
        if (dims[0] != 3 || dims[1] != 2)
            throw LibraryError("Bounds3DRef: 3D bounding box expected.");
    }

    inline T xmin() const { return (*this)[0]; }
    inline T xmax() const { return (*this)[1]; }
    inline T ymin() const { return (*this)[2]; }
    inline T ymax() const { return (*this)[3]; }
    inline T zmin() const { return (*this)[4]; }
    inline T zmax() const { return (*this)[5]; }

    inline T xDim() const { return (*this)[1] - (*this)[0] + 1; }
    inline T yDim() const { return (*this)[3] - (*this)[2] + 1; }
    inline T zDim() const { return (*this)[5] - (*this)[4] + 1; }

    inline CoordBBox toCoordBBox() const
    {
        return CoordBBox(xmin(), ymin(), zmin(), xmax(), ymax(), zmax());
    }

    inline bool isDegenerate() const {
        return xmin() > xmax() || ymin() > ymax() || zmin() > zmax();
    }
};

typedef Bounds3DRef<mint>    IntBounds3DRef;
typedef Bounds3DRef<double>  RealBounds3DRef;

template<typename T>
inline Bounds3DRef<T> makeBounds3D() { return makeMatrix<T>(3, 2); }


template<typename T>
class Bounds2DRef : public TensorRef<T>
{
public:
    Bounds2DRef(const TensorRef<T> &tr) : TensorRef<T>(tr)
    {
        if (TensorRef<T>::rank() != 2)
            throw LibraryError("Bounds2DRef: 2D bounding box expected.");
        const mint *dims = TensorRef<T>::dimensions();
        if (dims[0] != 2 || dims[1] != 2)
            throw LibraryError("Bounds2DRef: 2D bounding box expected.");
    }

    inline T xmin() const { return (*this)[0]; }
    inline T xmax() const { return (*this)[1]; }
    inline T ymin() const { return (*this)[2]; }
    inline T ymax() const { return (*this)[3]; }

    inline T xDim() const { return (*this)[1] - (*this)[0] + 1; }
    inline T yDim() const { return (*this)[3] - (*this)[2] + 1; }

    inline CoordBBox toCoordBBox() const
    {
        return CoordBBox(xmin(), ymin(), 0, xmax(), ymax(), 0);
    }

    inline bool isDegenerate() const {
        return xmin() > xmax() || ymin() > ymax();
    }
};

typedef Bounds2DRef<mint>    IntBounds2DRef;
typedef Bounds2DRef<double>  RealBounds2DRef;

template<typename T>
inline Bounds2DRef<T> makeBounds2D() { return makeMatrix<T>(2, 2); }


// ------------ initializers to convert mma types to vdb types ------------ //

template<typename T, typename VectorT>
inline VectorT toVDB(const VectorRef<T> mmavec)
{
    VectorT vec;
    for (int i = 0; i < vec.size; ++i)
        vec[i] = mmavec[i];

    return vec;
}

template<typename T, typename S>
inline S toVDB(const T &val) { return (S)val; }

template<typename T, typename VectorT>
inline VectorT toVDB(const MatrixRef<T> mmamat, const int i)
{
    VectorT vec;
    for (int j = 0; j < vec.size; ++j) {
        vec[j] = mmamat(i, j);
    }

    return vec;
}

template<typename T, typename S>
inline S toVDB(const VectorRef<T> mmavec, const int i) { return (S)mmavec[i]; }

} // namespace mma

#endif // OPENVDBLINK_OPENVDBCOMMON_HAS_BEEN_INCLUDED
