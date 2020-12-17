// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file GridAssetUrl.h

	\author Wil Braithwaite

	\date October 10, 2020

	\brief Class definition for handling Grid-Asset URLs.
*/

#pragma once

#include <string>

// The URL which contains a fragment which is a grid with an optional sequence.
// i.e.
// <scheme><authority><path><query>#<gridName>[<start>-<end>]
class GridAssetUrl
{
public:
    GridAssetUrl() = default;

    inline GridAssetUrl(const char* s)
    {
        assign(s);
    }

    inline GridAssetUrl(const std::string& s)
    {
        assign(s.c_str());
    }

    friend bool operator==(const GridAssetUrl& a, const GridAssetUrl& b);

    friend bool operator!=(const GridAssetUrl& a, const GridAssetUrl& b);

    inline operator bool() const
    {
        return !mPath.empty() || !mGridName.empty();
    }

    inline std::string        url() const { return mScheme + "://" + mPath; }
    inline std::string&       path() { return mPath; }
    inline std::string&       scheme() { return mScheme; }
    inline std::string&       gridName() { return mGridName; }
    inline const std::string& path() const { return mPath; }
    inline const std::string& scheme() const { return mScheme; }
    inline const std::string& gridName() const { return mGridName; }
    inline int                frameStart() const { return mFrameStart; }
    inline int                frameEnd() const { return mFrameEnd; }
    inline bool               isSequence() const { return mIsSequence; }
    inline int                frameStep() const { return mFrameStep; }

    std::string getSequencePath(int frame) const;
    std::string fullname() const;

    std::string updateUrlWithFrame(int frame) const;

private:
    void assign(const char* s);

    std::string mScheme;
    std::string mPath;
    std::string mGridName;
    bool        mIsSequence = false;
    int         mFrameStart = 0;
    int         mFrameEnd = -1;
    int         mFrameStep = 0;
};
