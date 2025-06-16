// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Loader.h

    \author Andrew Reidmeyer

    \brief  This file provides a compiler abstraction.
*/

#ifndef NANOVDB_PUTILS_LOADER_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_LOADER_H_HAS_BEEN_INCLUDED

// ------------------------------------------------ Loader -----------------------------------------------------------

#if defined(_WIN32)
#include <Windows.h>
static void* pnanovdb_load_library(const char* winName, const char* linuxName, const char* macName)
{
    return (void*)LoadLibraryA(winName);
}
static void* pnanovdb_get_proc_address(void* module, const char* name)
{
    return GetProcAddress((HMODULE)module, name);
}
static void pnanovdb_free_library(void* module)
{
    FreeLibrary((HMODULE)module);
}
static const char* pnanovdb_load_library_error()
{
    DWORD lastError = GetLastError();
    static char buf[1024];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, sizeof(buf), NULL);
    return buf;
}
#else
#include <dlfcn.h>
static void* pnanovdb_load_library(const char* winName, const char* linuxName, const char* macName)
{
#if defined(__APPLE__)
    void* module = dlopen(macName, RTLD_NOW);
#else
    void* module = dlopen(linuxName, RTLD_NOW);
#endif
    //if (!module)
    //{
    //    fprintf(stderr, "Module %s failed to load : %s\n", linuxName, dlerror());
    //}
    return module;
}
static void* pnanovdb_get_proc_address(void* module, const char* name)
{
    return dlsym(module, name);
}
static void pnanovdb_free_library(void* module)
{
    // if (!module)
    // {
    //     return;
    // }
    dlclose(module);
}
static const char* pnanovdb_load_library_error()
{
    return dlerror();
}
#endif

#endif
