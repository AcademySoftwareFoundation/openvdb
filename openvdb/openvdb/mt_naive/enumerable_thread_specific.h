
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file enumerable_thread_specific.h

#ifndef OPENVDB_ENUMERABLE_THREAD_SPECIFIC_HAS_BEEN_INCLUDED
#define OPENVDB_ENUMERABLE_THREAD_SPECIFIC_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <cassert>

//#include <tbb/enumerable_thread_specific.h>
#include <thread>
#include <shared_mutex>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

template <typename T, typename Values>
class enumerable_thread_specific_iterator_base
{
public:
    using Self = enumerable_thread_specific_iterator_base<T, Values>;

    enumerable_thread_specific_iterator_base(std::shared_mutex & lock, Values & locals, size_t i)
        : m_lock(lock)
        , m_locals(locals)
        , m_i(i)
    {}

    bool operator==(const Self & rhs) const
    {
        assert(&m_lock == &rhs.m_lock);
        assert(&m_locals == &rhs.m_locals);
        return m_i == rhs.m_i;
    }

    bool operator!=(const Self & rhs) const
    {
        assert(&m_lock == &rhs.m_lock);
        assert(&m_locals == &rhs.m_locals);
        return m_i != rhs.m_i;
    }

    bool operator!=(const T & rhs) const
    {
        assert(&m_lock == &rhs.m_lock);
        assert(&m_locals == &rhs.m_locals);

        std::shared_lock lock(m_lock);
        auto it = m_locals.begin()+m_i;
        return it->second != rhs;
    }

    Self & operator++()
    {
        ++m_i;
    }

    Self & operator++(int)
    {
        ++m_i;
    }

protected:
    std::shared_mutex & m_lock;
    Values & m_locals;
    size_t m_i;
};

template <typename T, typename Values>
class enumerable_thread_specific_iterator
    : public enumerable_thread_specific_iterator_base<T, Values>
{
public:
    using Base = enumerable_thread_specific_iterator_base<T, Values>;
    using Base::Base;

    T & operator*()
    {
        std::shared_lock lock(Base::m_lock);
        auto it = Base::m_locals.begin()+Base::m_i;
        return it->second;
    }
};

template <typename T, typename Values>
class enumerable_thread_specific_const_iterator
    : public enumerable_thread_specific_iterator_base<T, Values>
{
public:
    using Base = enumerable_thread_specific_iterator_base<T, Values>;
    using Base::Base;

    const T & operator*() const
    {
        std::shared_lock lock(Base::m_lock);
        auto it = Base::m_locals.begin()+Base::m_i;
        return it->second;
    }
};

template <typename T>
class enumerable_thread_specific
{
public:
    using Values = std::unordered_map<std::thread::id, T>;
    using iterator = enumerable_thread_specific_iterator<T, Values>;
    using const_iterator = enumerable_thread_specific_const_iterator<T, const Values>;

    iterator begin()
    {
        return iterator(m_lock, m_locals, 0);
    }

    iterator end()
    {
        std::shared_lock lock(m_lock);
        return iterator(m_lock, m_locals, m_locals.size());
    }

    const_iterator begin() const
    {
        return const_iterator(m_lock, m_locals, 0);
    }

    const_iterator end() const
    {
        std::shared_lock lock(m_lock);
        return const_iterator(m_lock, m_locals, m_locals.size());
    }

    bool empty()
    {
        std::shared_lock lock(m_lock);
        return m_locals.empty();
    }

    size_t size() const
    {
        std::shared_lock lock(m_lock);
        return m_locals.size();
    }

    void clear()
    {
        std::unique_lock lock(m_lock);
        m_locals.clear();
    }

    T& local(bool & exists)
    {
        std::thread::id id = std::this_thread::get_id();

        // The fast path, find the entry if it exists
        // and just return it
        {
            std::shared_lock lock(m_lock);
            typename Values::iterator it = m_locals.find(id);
            if( it != m_locals.end() )
            {
                exists = true;
                return it->second;
            }
        }

        // The slow path
        {
            std::unique_lock lock(m_lock);
            auto result = m_locals.emplace(id, T());
            exists = false;
            return result.first->second;
        }
    }

    T& local()
    {
        bool exists = false;
        return local(exists);
    }

    // combine_func_t has signature T(T,T) or T(const T&, const T&)
    template <typename combine_func_t>
    T combine(combine_func_t f_combine)
    {
        std::shared_lock lock(m_lock);
        return std::reduce( m_locals.begin(), m_locals.end(), f_combine );
    }

    // combine_func_t has signature void(T) or void(const T&)
    template <typename combine_func_t>
    void combine_each(combine_func_t f_combine)
  {
        std::shared_lock lock(m_lock);
        std::for_each( m_locals.begin(), m_locals.end(), f_combine );
    }

private:
    mutable std::shared_mutex m_lock;
    Values m_locals;
};

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_ENUMERABLE_THREAD_SPECIFIC_HAS_BEEN_INCLUDED
