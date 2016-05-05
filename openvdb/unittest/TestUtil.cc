///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#include <cppunit/extensions/HelperMacros.h>

#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <openvdb/Exceptions.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/util/PagedArray.h>

//#define BENCHMARK_PAGED_ARRAY

// For benchmark comparisons
#ifdef BENCHMARK_PAGED_ARRAY
#include <vector>// for std::vector
#include <deque>// for std::deque
#include <tbb/tbb.h>// for tbb::concurrent_vector
#endif

class TestUtil: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestUtil);
    CPPUNIT_TEST(testCpuTimer);
    CPPUNIT_TEST(testPagedArray);
    CPPUNIT_TEST_SUITE_END();

    void testCpuTimer();
    void testPagedArray();
    
    // Multi-threading T::push_back
    template<typename T>
    struct Functor {
        Functor(size_t begin, size_t end, size_t grainSize,
               T& _d, bool threaded = true) : d(&_d) {
            if (threaded) {
                tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, grainSize), *this);
            } else {
                (*this)(tbb::blocked_range<size_t>(begin, end));
            }
        }
        void operator()(const tbb::blocked_range<size_t>& r) const {
            for (size_t i=r.begin(), n=r.end(); i!=n; ++i) d->push_back(i);
        }
        T* d;
    };
    // Multi-threading T::ValueBuffer::push_back
    template<typename T>
    struct Functor2 {
        Functor2(size_t begin, size_t end, size_t grainSize,
                T& d, bool threaded = true) : buffer(d) {
            if (threaded) {
                tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, grainSize), *this);
            } else {
                (*this)(tbb::blocked_range<size_t>(begin, end));
            }
        }
        void operator()(const tbb::blocked_range<size_t>& r) const {
            for (size_t i=r.begin(), n=r.end(); i!=n; ++i) buffer.push_back(i);
        }
        mutable typename T::ValueBuffer buffer;
    };
    // Thread Local Storage version
    template<typename T>
    struct Functor3 {
        Functor3(size_t begin, size_t end, size_t grainSize,
                T& _pool, bool threaded = true) : pool(&_pool) {
            if (threaded) {
                tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, grainSize), *this);
            } else {
                (*this)(tbb::blocked_range<size_t>(begin, end));
            }
        }
        void operator()(const tbb::blocked_range<size_t>& r) const {
            typename T::reference buffer = pool->local();
            for (size_t i=r.begin(), n=r.end(); i!=n; ++i) buffer.push_back(i);
        }
        T* pool;
    };
    
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestUtil);

void
TestUtil::testCpuTimer()
{
    const int expected = 259, tolerance = 10;//milliseconds
    const tbb::tick_count::interval_t sec(expected/1000.0);
    
    openvdb::util::CpuTimer timer;
    tbb::this_tbb_thread::sleep(sec);
    const int actual1 = static_cast<int>(timer.delta());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual1, tolerance);
    tbb::this_tbb_thread::sleep(sec);
    const int actual2 = static_cast<int>(timer.delta());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2*expected, actual2, tolerance);
}

void
TestUtil::testPagedArray()
{
    
#ifdef BENCHMARK_PAGED_ARRAY
    const size_t problemSize = 2560000;
    openvdb::util::CpuTimer timer;
    std::cerr << "\nProblem size for benchmark: " << problemSize << std::endl;
#else
    const size_t problemSize = 256000;
#endif
    
    {//serial PagedArray::push_back (manual)
        openvdb::util::PagedArray<int, size_t(8)> d;
        CPPUNIT_ASSERT(d.isEmpty());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(8), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.capacity());
        
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.push_back(10));
        CPPUNIT_ASSERT_EQUAL(10, d[0]);
        CPPUNIT_ASSERT(!d.isEmpty());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.capacity());

        CPPUNIT_ASSERT_EQUAL(10, d.pop_back());
        CPPUNIT_ASSERT(d.isEmpty());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.capacity());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.push_back(10));
        
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(size_t(2), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.capacity());

        for (size_t i=2; i<d.pageSize(); ++i) CPPUNIT_ASSERT_EQUAL(i, d.push_back(int(i)));
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.capacity());

        for (int i=2, n=int(d.size()); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(d.pageSize()+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(2), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(2*d.pageSize(), d.capacity());
    }
    {//serial PagedArray::push_back (using Functor)
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("Serial test 1: Default page size");
#endif
        Functor<ArrayT> t(0, problemSize, ArrayT::pageSize(), d, false);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(size_t(10), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//parallel PagedArray::push_back
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel test 1: Default page size");
#endif
        Functor<ArrayT> t(0, problemSize, ArrayT::pageSize(), d);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(size_t(10), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort with a default page size");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

        CPPUNIT_ASSERT_EQUAL(problemSize, d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());


    }
    {//parallel PagedArray::push_back
        typedef openvdb::util::PagedArray<size_t, size_t(3)> ArrayT;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel test 1: Page size of only 8");
#endif
        Functor<ArrayT> t(0, problemSize, ArrayT::pageSize(), d);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(size_t(3), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
        
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort with a page size of only 8");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

        CPPUNIT_ASSERT_EQUAL(problemSize, d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
    }
#ifdef BENCHMARK_PAGED_ARRAY
    {//benchmark against a std::vector
        timer.start("std::vector");
        std::vector<size_t> v;
        for (size_t i=0; i<problemSize; ++i) v.push_back(i);
        timer.stop();
        CPPUNIT_ASSERT_EQUAL(problemSize, v.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, v[i]);
    }
    {//benchmark against a std::deque
        timer.start("std::deque");
        std::deque<size_t> d;
        for (size_t i=0; i<problemSize; ++i) d.push_back(i);
        timer.stop();
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());

        std::deque<int> d2;
        CPPUNIT_ASSERT_EQUAL(size_t(0), d2.size());
        d2.resize(1234);
        CPPUNIT_ASSERT_EQUAL(size_t(1234), d2.size());
    }
    {//benchmark against a tbb::concurrent_vector::push_back
        timer.start("serial tbb::concurrent_vector::push_back");
        tbb::concurrent_vector<size_t> v;
        for (size_t i=0; i<problemSize; ++i) v.push_back(i);
        timer.stop();
        CPPUNIT_ASSERT_EQUAL(problemSize, v.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, v[i]);

        v.clear();
        timer.start("parallel tbb::concurrent_vector::push_back");
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        Functor<tbb::concurrent_vector<size_t> > t(0, problemSize, ArrayT::pageSize(), v);
        timer.stop();
    }
#endif
    {//benchmark against a PagedArray::push_back
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("serial PagedArray::push_back");
#endif
        openvdb::util::PagedArray<size_t> d;
        for (size_t i=0; i<problemSize; ++i) d.push_back(i);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//benchmark against a PagedArray::push_back_unsafe
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("serial PagedArray::push_back_unsafe");
#endif
        openvdb::util::PagedArray<size_t, 10> d;
        for (size_t i=0; i<problemSize; ++i) d.push_back_unsafe(i);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//parallel push_back with resize and fill
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;

        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
        d.resize(problemSize);
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((problemSize-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        d.clear();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("serial PagedArray::push_back");
#endif
        Functor<ArrayT> t(0, problemSize, ArrayT::pageSize(), d, false);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif

        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
        
        d.clear();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel PagedArray::push_back");
#endif
        Functor<ArrayT> tmp1(0, problemSize, ArrayT::pageSize(), d, true);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif

        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        d.clear();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("serial PagedArray::ValueBuffer::push_back");
#endif
        Functor2<ArrayT> tmp2(0, problemSize, ArrayT::pageSize(), d, false);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif

        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        d.clear();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel PagedArray::ValueBuffer::push_back");
#endif
        Functor2<ArrayT> tmp3(0, problemSize, ArrayT::pageSize(), d, true);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        CPPUNIT_ASSERT_EQUAL(problemSize, d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL(size_t(1)+(problemSize>>d.log2PageSize()), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        const size_t v = 13;
        d.fill(v);
        for (size_t i=0, n=d.capacity(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(v, d[i]);
    }
    {//serial PagedArray::ValueBuffer::push_back
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
        {
            ArrayT::ValueBuffer vc(d);
            vc.push_back(1);
            vc.push_back(2);
            CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
            vc.flush();
            CPPUNIT_ASSERT_EQUAL(size_t(2), d.size());
            CPPUNIT_ASSERT_EQUAL(size_t(1), d[0]);
            CPPUNIT_ASSERT_EQUAL(size_t(2), d[1]);
        }
        CPPUNIT_ASSERT_EQUAL(size_t(2), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d[0]);
        CPPUNIT_ASSERT_EQUAL(size_t(2), d[1]);
    }
    {//parallel PagedArray::push_back followed by parallel sort
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel PagedArray::push_back");
#endif
        Functor<ArrayT> t(0, problemSize, ArrayT::pageSize(), d);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        // Not guaranteed to pass
        //size_t unsorted = 0;
        //for (size_t i=0, n=d.size(); i<n; ++i) unsorted += i != d[i];
        //CPPUNIT_ASSERT( unsorted > 0 );

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//serial PagedArray::ValueBuffer::push_back followed by parallel sort
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("serial PagedArray::ValueBuffer::push_back");
#endif
        Functor2<ArrayT> t(0, problemSize, ArrayT::pageSize(), d, false);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
        size_t unsorted = 0;
        for (size_t i=0, n=d.size(); i<n; ++i) unsorted += i != d[i];
        CPPUNIT_ASSERT_EQUAL(size_t(0), unsorted);

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//parallel PagedArray::ValueBuffer::push_back followed by parallel sort
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel PagedArray::ValueBuffer::push_back");
#endif
        Functor2<ArrayT> t(0, problemSize, ArrayT::pageSize(), d, true);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        // Not guaranteed to pass
        //size_t unsorted = 0;
        //for (size_t i=0, n=d.size(); i<n; ++i) unsorted += i != d[i];
        //CPPUNIT_ASSERT( unsorted > 0 );

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel inverse sort");
#endif
        d.invSort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size()-1; i<=n; ++i) CPPUNIT_ASSERT_EQUAL(n-i, d[i]);
    }
    {//thread-local-storage PagedArray::ValueBuffer::push_back followed by parallel sort
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d;        

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel TLS PagedArray::ValueBuffer::push_back");
#endif
        ArrayT::ValueBuffer tmp(d);
        typedef tbb::enumerable_thread_specific<ArrayT::ValueBuffer> BufferT;
        BufferT buffer(tmp);
        Functor3<BufferT> t(0, problemSize, ArrayT::pageSize(), buffer, true);
        for (BufferT::iterator i = buffer.begin(); i != buffer.end(); ++i) i->flush();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        //std::cerr << "Number of threads for TLS = " << (buffer.end()-buffer.begin()) << std::endl;
        //d.print();
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        // Not guaranteed to pass
        //size_t unsorted = 0;
        //for (size_t i=0, n=d.size(); i<n; ++i) unsorted += i != d[i];
        //CPPUNIT_ASSERT( unsorted > 0 );

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);        
    }
    {//parallel PagedArray::merge followed by parallel sort
        typedef openvdb::util::PagedArray<size_t> ArrayT;
        ArrayT d, d2;
        
        Functor2<ArrayT> t1(0, problemSize, ArrayT::pageSize(), d, true);
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
        CPPUNIT_ASSERT(!d.isPartiallyFull());
        d.push_back(problemSize);
        CPPUNIT_ASSERT(d.isPartiallyFull());

        Functor2<ArrayT> t2(problemSize+1, 2*problemSize+1, ArrayT::pageSize(), d2, true);
        //for (size_t i=d.size(), n=i+problemSize; i<n; ++i) d2.push_back(i);
        CPPUNIT_ASSERT(!d2.isPartiallyFull());
        CPPUNIT_ASSERT_EQUAL(problemSize, d2.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d2.log2PageSize(), d2.pageSize());
        CPPUNIT_ASSERT_EQUAL((d2.size()-1)>>d2.log2PageSize(), d2.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d2.pageCount()*d2.pageSize(), d2.capacity());

        //d.print();
        //d2.print();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel PagedArray::merge");
#endif
        d.merge(d2);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT(d.isPartiallyFull());
        
        //d.print();
        //d2.print();
        CPPUNIT_ASSERT_EQUAL(2*problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(size_t(0), d2.size());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d2.pageCount());

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort of merged array");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//examples in doxygen
        {//1
            openvdb::util::PagedArray<int> array;
            for (int i=0; i<100000; ++i) array.push_back(i);
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//2A
            openvdb::util::PagedArray<int> array;    
            openvdb::util::PagedArray<int>::ValueBuffer buffer(array);
            for (int i=0; i<100000; ++i) buffer.push_back(i);    
            buffer.flush();
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//2B
            openvdb::util::PagedArray<int> array;
            {//local scope of a single thread   
                openvdb::util::PagedArray<int>::ValueBuffer buffer(array);
                for (int i=0; i<100000; ++i) buffer.push_back(i);    
            }
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//3A
            openvdb::util::PagedArray<int> array;
            array.resize(100000);
            for (int i=0; i<100000; ++i) array[i] = i;
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//3B
            typedef openvdb::util::PagedArray<int> ArrayT;
            ArrayT array;
            array.resize(100000);
            for (ArrayT::Iterator i=array.begin(); i!=array.end(); ++i) *i = int(i.pos());
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
    }
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
