#pragma once

#include "../Utilities/metadata.h"


/* OpenVDBGrid public member function list

bool getBooleanMetadata(const char* key)

mint getIntegerMetadata(const char* key)

double getRealMetadata(const char* key)

const char * getStringMetadata(const char* key)

void setBooleanMetadata(const char* key, bool value)

void setStringMetadata(const char* key, const char* value)

void setDescription(const char* description)

*/


//////////// OpenVDBScalarGrid public member function definitions

template<typename V>
bool
OpenVDBGrid<V>::getBooleanMetadata(const char* key) const
{
    const string key_string(key);
    mma::disownString(key);
    
    openvdbmma::metadata::GridMetadata<wlGridType> meta(grid());
    const bool bval = meta.template getMetadata<bool>(key_string);
    
    return bval;
}

template<typename V>
mint
OpenVDBGrid<V>::getIntegerMetadata(const char* key) const
{
    const string key_string(key);
    mma::disownString(key);
    
    openvdbmma::metadata::GridMetadata<wlGridType> meta(grid());
    const mint ival = meta.template getMetadata<mint>(key_string);
    
    return ival;
}

template<typename V>
double
OpenVDBGrid<V>::getRealMetadata(const char* key) const
{
    const string key_string(key);
    mma::disownString(key);
    
    openvdbmma::metadata::GridMetadata<wlGridType> meta(grid());
    const double rval = meta.template getMetadata<float>(key_string);
    
    return rval;
}

template<typename V>
const char*
OpenVDBGrid<V>::getStringMetadata(const char* key)
{
    const string key_string(key);
    mma::disownString(key);
    
    openvdbmma::metadata::GridMetadata<wlGridType> meta(grid());
    const string sval = meta.template getMetadata<string>(key_string);
    
    //Let the class handle memory management when passing a string to WL
    return WLString(sval);
}

template<typename V>
void
OpenVDBGrid<V>::setBooleanMetadata(const char* key, bool value)
{
    const string key_string(key);
    mma::disownString(key);
    
    openvdbmma::metadata::GridMetadata<wlGridType> meta(grid());
    meta.template setMetadata<bool>(key_string, value);
}

template<typename V>
void
OpenVDBGrid<V>::setStringMetadata(const char* key, const char* value)
{
    const string key_string(key);
    mma::disownString(key);
    
    const string value_string(value);
    mma::disownString(value);
    
    openvdbmma::metadata::GridMetadata<wlGridType> meta(grid());
    meta.template setMetadata<string>(key_string, value_string);
}

template<typename V>
void
OpenVDBGrid<V>::setDescription(const char* description)
{
    const string description_string(description);
    mma::disownString(description);
    
    openvdbmma::metadata::GridMetadata<wlGridType> meta(grid());
    meta.template setMetadata<string>(META_DESCRIPTION, description_string);
}
