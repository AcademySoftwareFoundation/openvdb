// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_GLUETENSORS_HAS_BEEN_INCLUDED
#define OPENVDBLINK_GLUETENSORS_HAS_BEEN_INCLUDED

#include "OpenVDBCommon.h"
#include "LTemplate.h"

#include <openvdb/openvdb.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vector>

namespace openvdbmma {
namespace glue {


// ------------ Scalar classes ------------ //

// scalar

template<typename M>
class GlueScalar0 {

public:

    template<typename T>
    GlueScalar0(const T val)
    : mScalar((M)val)
    {
    }

    inline M mmaData() const
    {
        return mScalar;
    }

private:

    M mScalar;

}; // GlueScalar0


// vector <--> scalar

template<typename M, int m>
class GlueScalar1 {

public:

    GlueScalar1(const int n)
    : mVec(mma::VectorRef<M>(n))
    {
    }

    template<typename VectorT>
    GlueScalar1(const VectorT &vec)
    : mVec(mma::makeVector<M>(m))
    {
        for (int i = 0; i < m; ++i)
            mVec[i] = (M)vec[i];
    }

    inline mma::VectorRef<M> mmaData() const
    {
        return mVec;
    }

private:

    mma::VectorRef<M> mVec;

}; // GlueScalar1


// ------------ Vector classes ------------ //

// vector of scalars

template<typename M>
class GlueVector0 {

public:

    GlueVector0(const int n)
    : mVec(mma::makeVector<M>(n))
    {
    }

    template<typename T>
    GlueVector0(const mma::VectorRef<T> &vec)
    : mVec(mma::makeVector<M>(vec.size()))
    {
        for (int i = 0; i < vec.size(); ++i)
            mVec[i] = (M)vec[i];
    }

    template<typename S>
    GlueVector0(const std::vector<S> &vec)
    : mVec(mma::makeVector<M>(vec.size()))
    {
        tbb::parallel_for(
            tbb::blocked_range<mint>(0, vec.size()),
            [&](tbb::blocked_range<mint> rng)
            {
                for(mint i = rng.begin(); i < rng.end(); ++i) {
                    mVec[i] = (M)vec[i];
                }
            }
        );
    }

    template<typename VectorT>
    GlueVector0(const VectorT &vec)
    : mVec(mma::makeVector<M>(vec.size))
    {
        for (int i = 0; i < vec.size; ++i)
            mVec[i] = (M)vec[i];
    }

    inline mma::VectorRef<M> mmaData() const
    {
        return mVec;
    }

    template<typename T>
    inline void setValue(const int i, const T val)
    {
        mVec[i] = (M)val;
    }

private:

    mma::VectorRef<M> mVec;

}; // GlueVector0


// vector of vectors <--> matrix

template<typename M, int m>
class GlueVector1 {

public:

    GlueVector1(const int n)
    : mMat(mma::makeMatrix<M>(n, m))
    {
    }

    template<typename T>
    GlueVector1(const mma::MatrixRef<T> &mat)
    : mMat(mma::makeMatrix<M>(mat.rows(), m))
    {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < m; ++j)
                mMat(i, j) = (M)mat(i, j);
    }

    template<typename VectorT>
    GlueVector1(const std::vector<VectorT> &mat)
    : mMat(mma::makeMatrix<M>(mat.size(), m))
    {
        tbb::parallel_for(
            tbb::blocked_range<mint>(0, mat.size()),
            [&](tbb::blocked_range<mint> rng)
            {
                for(mint i = rng.begin(); i < rng.end(); ++i) {
                    for (int j = 0; j < m; ++j)
                        mMat(i, j) = (M)mat[i][j];
                }
            }
        );
    }

    inline mma::MatrixRef<M> mmaData() const
    {
        return mMat;
    }

    template<typename VectorT>
    inline void setValue(const int i, const VectorT vec)
    {
        for (int j = 0; j < mMat.cols(); ++j)
            mMat(i, j) = (M)vec[j];
    }

    template<typename T>
    inline void setValue(const int i, const int j, const T val)
    {
        mMat(i, j) = (M)val;
    }

private:

    mma::MatrixRef<M> mMat;

}; // GlueVector1


// ------------ Matrix classes ------------ //

// matrix of scalars

template<typename M>
class GlueMatrix0 {

public:

    GlueMatrix0(const int n, const int m)
    : mMat(mma::makeMatrix<M>(n, m))
    {
    }

    template<typename T>
    GlueMatrix0(const mma::MatrixRef<T> &mat)
    : mMat(mma::makeMatrix<M>(mat.rows(), mat.cols()))
    {
        tbb::parallel_for(
            tbb::blocked_range<mint>(0, mat.rows()),
            [&](tbb::blocked_range<mint> rng)
            {
                for(mint i = rng.begin(); i < rng.end(); ++i) {
                    for (int j = 0; j < mat.cols(); ++j)
                        mMat(i, j) = (M)mat(i, j);
                }
            }
        );
    }

    template<typename VectorT>
    GlueMatrix0(const std::vector<VectorT> &vec)
    : mMat(mma::makeMatrix<M>(vec.size(), vec[0].size))
    {
        const int m = vec[0].size;

        tbb::parallel_for(
            tbb::blocked_range<mint>(0, vec.size()),
            [&](tbb::blocked_range<mint> rng)
            {
                for(mint i = rng.begin(); i < rng.end(); ++i) {
                    for(mint j = 0; j < m; ++j)
                        mMat(i, j) = vec[i][j];
                }
            }
        );
    }

    inline mma::MatrixRef<M> mmaData() const
    {
        return mMat;
    }

    template<typename VectorT>
    inline void setValue(const int i, const VectorT vec)
    {
        for (int j = 0; j < vec.size; ++j)
            mMat(i, j) = (M)vec[j];
    }

    template<typename T>
    inline void setValue(const int i, const int j, const T val)
    {
        mMat(i, j) = (M)val;
    }

private:

    mma::MatrixRef<M> mMat;

}; // GlueMatrix0


// matrix of vectors <--> cube

template<typename M, int p>
class GlueMatrix1 {

public:

    GlueMatrix1(const int n, const int m)
    : mCube(mma::makeCube<M>(n, m, p))
    {
    }

    template<typename T>
    GlueMatrix1(const mma::CubeRef<T> &cube)
    : mCube(mma::makeCube<M>(cube.slices(), cube.rows(), cube.cols()))
    {
        tbb::parallel_for(
            tbb::blocked_range<mint>(0, cube.slices()),
            [&](tbb::blocked_range<mint> rng)
            {
                for(mint i = rng.begin(); i < rng.end(); ++i) {
                    for (int j = 0; j < cube.rows(); ++j)
                        for (int k = 0; k < cube.cols(); ++k)
                            mCube(i, j, k) = (M)cube(i, j, k);
                }
            }
        );
    }

    inline mma::CubeRef<M> mmaData() const
    {
        return mCube;
    }

    template<typename VectorT>
    inline void setValue(const int i, const int j, const VectorT vec)
    {
        for (int k = 0; k < mCube.cols(); ++k)
            mCube(i, j, k) = (M)vec[k];
    }

    template<typename T>
    inline void setValue(const int i, const int j, const int k, const T val)
    {
        mCube(i, j, k) = (M)val;
    }

private:

    mma::CubeRef<M> mCube;

}; // GlueMatrix1

// ------------ Cube classes ------------ //

// cube of scalars

template<typename M>
class GlueCube0 {

public:

    GlueCube0(const int n, const int m, const int p)
    : mCube(mma::makeCube<M>(n, m, p))
    {
    }

    template<typename T>
    GlueCube0(const mma::CubeRef<T> &cube)
    : mCube(mma::makeCube<M>(cube.slices(), cube.rows(), cube.cols()))
    {
        tbb::parallel_for(
            tbb::blocked_range<mint>(0, cube.slices()),
            [&](tbb::blocked_range<mint> rng)
            {
                for(mint i = rng.begin(); i < rng.end(); ++i) {
                    for (int j = 0; j < cube.rows(); ++j)
                        for (int k = 0; k < cube.cols(); ++k)
                            mCube(i, j, k) = (M)cube(i, j, k);
                }
            }
        );
    }

    inline mma::CubeRef<M> mmaData() const
    {
        return mCube;
    }

    template<typename VectorT>
    inline void setValue(const int i, const int j, const VectorT vec)
    {
        for (int k = 0; k < mCube.cols(); ++k)
            mCube(i, j, k) = (M)vec[k];
    }

    template<typename T>
    inline void setValue(const int i, const int j, const int k, const T val)
    {
        mCube(i, j, k) = (M)val;
    }

private:

    mma::CubeRef<M> mCube;

}; // GlueCube0


// cube of vectors <--> tesseract

template<typename M, int q>
class GlueCube1 {

public:

    GlueCube1(const int n, const int m, const int p)
    : mTess(mma::makeTesseract<M>(n, m, p, q))
    {
    }

    inline mma::TensorRef<M> mmaData() const
    {
        return mTess;
    }

    template<typename VectorT>
    inline void setValue(const int i, const int j, const int k, const VectorT vec)
    {
        for (int l = 0; l < vec.size; ++l)
            mTess(i, j, k, l) = (M)vec[l];
    }

    template<typename T>
    inline void setValue(const int i, const int j, const int k, const int l, const T val)
    {
        mTess(i, j, k, l) = (M)val;
    }

private:

    mma::TesseractRef<M> mTess;

}; // GlueCube1

} // namespace openvdbmma
} // namespace glue

#endif // OPENVDBLINK_GLUETENSORS_HAS_BEEN_INCLUDED
