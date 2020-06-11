///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
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

/// @file compiler/CustomData.h
///
/// @authors Nick Avramoussis, Francisco Gochez
///
/// @brief  Access to the CustomData class which can provide custom user
///   user data to the OpenVDB AX Compiler.
///

#ifndef OPENVDB_AX_COMPILER_CUSTOM_DATA_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_CUSTOM_DATA_HAS_BEEN_INCLUDED

#include <openvdb_ax/ast/Literals.h>

#include <openvdb/Metadata.h>
#include <openvdb/Types.h>

#include <unordered_map>
#include <memory>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

///  @brief The custom data class is a simple container for named openvdb metadata.  Its primary use
///         case is passing arbitrary "external" data to an AX executable object when calling
///         Compiler::compile. For example, it is the mechanism by which we pass data held inside of a
///         parent DCC to executable AX code.
class CustomData
{
public:

    using Ptr = std::shared_ptr<CustomData>;
    using ConstPtr = std::shared_ptr<const CustomData>;
    using UniquePtr = std::unique_ptr<CustomData>;

    CustomData() : mData() {}

    static UniquePtr create()
    {
        UniquePtr data(new CustomData);
        return data;
    }

    /// @brief Reset the custom data. This will clear and delete all previously added data.
    /// @note  When used the Compiler::compile method, this should not be used prior to executing
    ///        the built executable object
    inline void reset()
    {
        mData.clear();
    }

    /// @brief  Checks whether or not data of given name has been inserted
    inline bool
    hasData(const Name& name)
    {
        const auto iter = mData.find(name);
        return (iter != mData.end());
    }

    /// @brief  Checks whether or not data of given name and type has been inserted
    template <typename TypedDataCacheT>
    inline bool
    hasData(const Name& name)
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) return false;
        const TypedDataCacheT* const typed =
            dynamic_cast<const TypedDataCacheT* const>(iter->second.get());
        return typed != nullptr;
    }

    /// @brief  Retrieves a const pointer to data of given name.  If it does not
    ///         exist, returns nullptr
    inline const Metadata::ConstPtr
    getData(const Name& name) const
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) return Metadata::ConstPtr();
        return iter->second;
    }

    /// @brief   Retrieves a const pointer to data of given name and type.  If it does not
    ///          exist, returns nullptr
    /// @param   name Name of the data entry
    /// @returns Object of given type and name.  If the type does not match, nullptr is returned.
    template <typename TypedDataCacheT>
    inline const TypedDataCacheT*
    getData(const Name& name) const
    {
        Metadata::ConstPtr data = getData(name);
        if (!data) return nullptr;
        const TypedDataCacheT* const typed =
            dynamic_cast<const TypedDataCacheT* const>(data.get());
        return typed;
    }

    /// @brief  Retrieves or inserts typed metadata. If thedata exists, it is dynamic-casted to the
    ///         expected type, which may result in a nullptr. If the data does not exist it is guaranteed
    ///         to be inserted and returned. The value of the inserted data can then be modified
    template <typename TypedDataCacheT>
    inline TypedDataCacheT*
    getOrInsertData(const Name& name)
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) {
            Metadata::Ptr data(new TypedDataCacheT());
            mData[name] = data;
            return static_cast<TypedDataCacheT* const>(data.get());
        }
        else {
            return dynamic_cast<TypedDataCacheT* const>(iter->second.get());
        }
    }

    /// @brief  Inserts data of specified type with given name.
    /// @param  name Name of the data
    /// @param  data Shared pointer to the data
    /// @note   If an entry of the given name already exists, will copy the data into the existing
    ///         entry rather than overwriting the pointer
    template <typename TypedDataCacheT>
    inline void
    insertData(const Name& name,
               const typename TypedDataCacheT::Ptr data)
    {
        if (hasData(name)) {
            TypedDataCacheT* const dataToSet =
                getOrInsertData<TypedDataCacheT>(name);
            if (!dataToSet) {
                OPENVDB_THROW(TypeError, "Custom data \"" + name + "\" already exists with a different type.");
            }
            dataToSet->value() = data->value();
        }
        else {
            mData[name] = data->copy();
        }
    }

    /// @brief  Inserts data with given name.
    /// @param  name Name of the data
    /// @param  data The metadata
    /// @note   If an entry of the given name already exists, will copy the data into the existing
    ///         entry rather than overwriting the pointer
    inline void
    insertData(const Name& name,
               const Metadata::Ptr data)
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) {
            mData[name] = data;
        }
        else {
            iter->second->copy(*data);
        }
    }

private:
    std::unordered_map<Name, Metadata::Ptr> mData;
};


struct AXStringMetadata : public StringMetadata
{
    using Ptr = openvdb::SharedPtr<AXStringMetadata>;
    using ConstPtr = openvdb::SharedPtr<const AXStringMetadata>;

    AXStringMetadata(const std::string& string)
        : StringMetadata(string)
        , mData()
    {
        this->initialize();
    }

    // delegate, ensure valid string initialization
    AXStringMetadata() : AXStringMetadata("") {}
    AXStringMetadata(const AXStringMetadata& other)
        : StringMetadata(other)
        , mData()
    {
        this->initialize();
    }

    ~AXStringMetadata() override {}

    openvdb::Metadata::Ptr copy() const override {
        openvdb::Metadata::Ptr metadata(new AXStringMetadata());
        metadata->copy(*this);
        return metadata;
    }

    void copy(const openvdb::Metadata& other) override {
        const AXStringMetadata* t = dynamic_cast<const AXStringMetadata*>(&other);
        if (t == nullptr) OPENVDB_THROW(openvdb::TypeError, "Incompatible type during copy");
        this->StringMetadata::setValue(t->StringMetadata::value());
        this->initialize();
    }

    void setValue(const std::string& string)
    {
        this->StringMetadata::setValue(string);
        this->initialize();
    }

    ax::AXString& value() { return mData; }
    const ax::AXString& value() const { return mData; }

protected:

    void readValue(std::istream& is, openvdb::Index32 size) override {
        StringMetadata::readValue(is, size);
        this->initialize();
    }

private:
    void initialize()
    {
        mData.ptr = StringMetadata::value().c_str();
        mData.size = StringMetadata::value().size();
    }

    ax::AXString mData;
};


}
}
}

#endif // OPENVDB_AX_COMPILER_CUSTOM_DATA_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
