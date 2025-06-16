// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compiler/TrackedFileSystem.h

    \author Petra Hapalova

    \brief
*/

#pragma once

#include <slang.h>
#include <slang-com-ptr.h>
#include <vector>
#include <string>
#include <fstream>
#include <atomic>

namespace pnanovdb_compiler
{
    class MutableBlob : public ISlangBlob
    {
    public:
        MutableBlob(const void* data, size_t size)
            : m_size(size), m_data(new char[size])
        {
            std::memcpy(m_data, data, size);
        }

        virtual ~MutableBlob()
        {
            delete[] m_data;
        }

        virtual void const* SLANG_MCALL getBufferPointer() override
        {
            return m_data;
        }

        virtual size_t SLANG_MCALL getBufferSize() override
        {
            return m_size;
        }

        // ISlangUnknown methods
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject) override
        {
            if (uuid == ISlangUnknown::getTypeGuid() || uuid == ISlangBlob::getTypeGuid())
            {
                *outObject = static_cast<ISlangBlob*>(this);
                addRef();
                return SLANG_OK;
            }
            *outObject = nullptr;
            return SLANG_E_NO_INTERFACE;
        }

        virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override
        {
            return ++m_refCount;
        }

        virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override
        {
            uint32_t newCount = --m_refCount;
            if (newCount == 0) delete this;
            return newCount;
        }
    private:
        size_t m_size;
        char* m_data;
        std::atomic<uint32_t> m_refCount{ 1 };
    };

    class TrackedFileSystem : public ISlangFileSystem
    {
    public:
        // ISlangUnknown methods
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject) SLANG_OVERRIDE
        {
            if (uuid == ISlangUnknown::getTypeGuid())
            {
                *outObject = static_cast<ISlangUnknown*>(this);
                addRef();
                return SLANG_OK;
            }
            if (uuid == ISlangFileSystem::getTypeGuid())
            {
                *outObject = static_cast<ISlangFileSystem*>(this);
                addRef();
                return SLANG_OK;
            }
            return SLANG_E_NO_INTERFACE;
        }

        virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE
        {
            return 1;
        }

        virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE
        {
            return 1;
        }

        // ISlangCastable method
        virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& uuid) SLANG_OVERRIDE
        {
            if (uuid == ISlangFileSystem::getTypeGuid())
            {
                return static_cast<ISlangFileSystem*>(this);
            }
            if (uuid == ISlangCastable::getTypeGuid())
            {
                return static_cast<ISlangCastable*>(this);
            }
            if (uuid == ISlangUnknown::getTypeGuid())
            {
                return static_cast<ISlangUnknown*>(this);
            }
            return nullptr;
        }

        // ISlangFileSystem methods
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(char const* path, ISlangBlob** outBlob) SLANG_OVERRIDE
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open())
            {
                return SLANG_FAIL;
            }

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> buffer(size);
            if (!file.read(buffer.data(), size))
            {
                return SLANG_FAIL;
            }

            if (!outBlob)
            {
                return SLANG_FAIL;
            }

            trackedFiles_.push_back(path);

            *outBlob = new MutableBlob(buffer.data(), buffer.size());
            return SLANG_OK;
        }

        const std::vector<std::string>& getTrackedFiles() const
        {
            return trackedFiles_;
        }

        void clearTrackedFiles()
        {
            trackedFiles_.clear();
        }

    private:
        std::vector<std::string> trackedFiles_;
    };

} // namespace pnanovdb_compiler
