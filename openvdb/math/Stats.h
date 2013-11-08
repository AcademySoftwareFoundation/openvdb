///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file Stats.h
///
/// @brief Classes to compute statistics and histograms

#ifndef OPENVDB_MATH_STATS_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_STATS_HAS_BEEN_INCLUDED

#include <iosfwd> // for ostringstream
#include <openvdb/version.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include "Math.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @brief This class computes statistics (minimum value, maximum
/// value, mean, variance and standard deviation) of a population
/// of floating-point values.
///
/// @details variance = Mean[ (X-Mean[X])^2 ] = Mean[X^2] - Mean[X]^2,
///          standard deviation = sqrt(variance)
///
/// @note This class employs incremental computation and double precision.
class Stats
{
public:
    Stats(): mSize(0), mAvg(0.0), mAux(0.0),
        mMin(std::numeric_limits<double>::max()), mMax(-mMin) {}

    /// Add a single sample.
    void add(double val)
    {
        mSize++;
        mMin = std::min<double>(val, mMin);
        mMax = std::max<double>(val, mMax);
        const double delta = val - mAvg;
        mAvg += delta/double(mSize);
        mAux += delta*(val - mAvg);
    }

    /// Add @a n samples with constant value @a val.
    void add(double val, uint64_t n)
    {
        mMin  = std::min<double>(val, mMin);
        mMax  = std::max<double>(val, mMax);
        const double denom = 1.0/double(mSize + n);
        const double delta = val - mAvg;
        mAvg += denom*delta*n;
        mAux += denom*delta*delta*mSize*n;
        mSize += n;
    }

    /// Add the samples from the other Stats instance.
    void add(const Stats& other)
    {
        if (other.mSize > 0) {
            mMin  = std::min<double>(mMin, other.mMin);
            mMax  = std::max<double>(mMax, other.mMax);
            const double denom = 1.0/double(mSize + other.mSize);
            const double delta = other.mAvg - mAvg;
            mAvg += denom*delta*other.mSize;
            mAux += other.mAux + denom*delta*delta*mSize*other.mSize;
            mSize += other.mSize;
        }
    }

    /// Return the size of the population, i.e., the total number of samples.
    inline uint64_t size() const { return mSize; }

    /// Return the minimum value.
    inline double min() const { return mMin; }

    /// Return the maximum value.
    inline double max() const { return mMax; }

    //@{
    /// Return the  arithmetic mean, i.e. average, value.
    inline double avg()  const { return mAvg; }
    inline double mean() const { return mAvg; }
    //@}

    //@{
    /// @brief Return the population variance.
    /// @note The unbiased sample variance = population variance *
    //num/(num-1)
    inline double var()      const { return mSize<2 ? 0.0 : mAux/double(mSize); }
    inline double variance() const { return this->var(); }
    //@}

    //@{
    /// @brief Return the standard deviation (=Sqrt(variance)) as
    /// defined from the (biased) population variance.
    inline double std()    const { return sqrt(this->var()); }
    inline double stdDev() const { return this->std(); }
    //@}

    /// @brief Print statistics to the specified output stream.
    void print(const std::string &name= "", std::ostream &strm=std::cout, int precision=3) const
    {
        // Write to a temporary string stream so as not to affect the state
        // (precision, field width, etc.) of the output stream.
        std::ostringstream os;
        os << std::setprecision(precision) << std::setiosflags(std::ios::fixed);
        os << "Statistics ";
        if (!name.empty()) os << "for \"" << name << "\" ";
        if (mSize>0) {
            os << "with " << mSize << " samples:\n"
               << "  Min=" << mMin
               << ", Max=" << mMax
               << ", Ave=" << mAvg
               << ", Std=" << this->stdDev()
               << ", Var=" << this->variance() << std::endl;
        } else {
            os << ": no samples were added." << std::endl;
        }
        strm << os.str();
    }

private:
    uint64_t mSize;
    double mAvg, mAux, mMin, mMax;
}; // end Stats


////////////////////////////////////////


/// @brief This class computes a histogram, with a fixed interval width,
/// of a population of floating-point values.
class Histogram
{
public:
    /// Construct with given minimum and maximum values and the given bin count.
    Histogram(double min, double max, size_t numBins = 10)
        : mSize(0), mMin(min), mMax(max+1e-10),
          mDelta(double(numBins)/(max-min)), mBins(numBins)
    {
        assert(numBins > 1);
        assert(mMax-mMin > 1e-10);
        for (size_t i=0; i<numBins; ++i) mBins[i]=0;
    }

    /// @brief Construct with the given bin count and with minimum and maximum values
    /// taken from a Stats object.
    Histogram(const Stats& s, size_t numBins = 10):
        mSize(0), mMin(s.min()), mMax(s.max()+1e-10),
        mDelta(double(numBins)/(mMax-mMin)), mBins(numBins)
    {
        assert(numBins > 1);
        assert(mMax-mMin > 1e-10);
        for (size_t i=0; i<numBins; ++i) mBins[i]=0;
    }

    /// @brief Add @a n samples with constant value @a val, provided that the
    /// @a val falls within this histogram's value range.
    /// @return @c true if the sample value falls within this histogram's value range.
    inline bool add(double val, uint64_t n = 1)
    {
        if (val<mMin || val>mMax) return false;
        mBins[size_t(mDelta*(val-mMin))] += n;
        mSize += n;
        return true;
    }

    /// @brief Add all the contributions from the other histogram, provided that
    /// it has the same configuration as this histogram.
    bool add(const Histogram& other)
    {
        if (!isApproxEqual(mMin, other.mMin) || !isApproxEqual(mMax, other.mMax) ||
            mBins.size() != other.mBins.size()) return false;
        for (size_t i=0, e=mBins.size(); i!=e; ++i) mBins[i] += other.mBins[i];
        mSize += other.mSize;
        return true;
    }

    /// Return the number of bins in this histogram.
    inline size_t numBins() const { return mBins.size(); }
    /// Return the lower bound of this histogram's value range.
    inline double min() const { return mMin; }
    /// Return the upper bound of this histogram's value range.
    inline double max() const { return mMax; }
    /// Return the minimum value in the <i>n</i>th bin.
    inline double min(int n) const { return mMin+n/mDelta; }
    /// Return the maximum value in the <i>n</i>th bin.
    inline double max(int n) const { return mMin+(n+1)/mDelta; }
    /// Return the number of samples in the <i>n</i>th bin.
    inline uint64_t count(int n) const { return mBins[n]; }
    /// Return the population size, i.e., the total number of samples.
    inline uint64_t size() const { return mSize; }

    /// Print the histogram to the specified output stream.
    void print(const std::string& name = "", std::ostream& strm = std::cout) const
    {
        // Write to a temporary string stream so as not to affect the state
        // (precision, field width, etc.) of the output stream.
        std::ostringstream os;
        os << std::setprecision(6) << std::setiosflags(std::ios::fixed) << std::endl;
        os << "Histogram ";
        if (!name.empty()) os << "for \"" << name << "\" ";
        if (mSize > 0) {
            os << "with " << mSize << " samples:\n";
            os << "==============================================================\n";
            os << "||  #   |       Min      |       Max      | Frequency |  %  ||\n";
            os << "==============================================================\n";
            for (size_t i=0, e=mBins.size(); i!=e; ++i) {
                os << "|| " << std::setw(4) << i   << " | " << std::setw(14) << this->min(i) << " | "
                   << std::setw(14) << this->max(i) << " | " << std::setw(9) << mBins[i]     << " | "
                   << std::setw(3) << (100*mBins[i]/mSize) << " ||\n";
            }
            os << "==============================================================\n";
        } else {
            os << ": no samples were added." << std::endl;
        }
        strm << os.str();
    }

private:
    uint64_t mSize;
    double mMin, mMax, mDelta;
    std::vector<uint64_t> mBins;
};

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_STATS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
