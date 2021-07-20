// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file GridAssetUrl.cpp
	\brief Class for handling Grid-Asset URLs.
*/

#include "GridAssetUrl.h"
#include "StringUtils.h"
#include <algorithm>
#include <sstream>

bool operator==(const GridAssetUrl& a, const GridAssetUrl& b)
{
    return a.fullname() == b.fullname();
}

bool operator!=(const GridAssetUrl& a, const GridAssetUrl& b)
{
    return a.fullname() != b.fullname();
}

std::string GridAssetUrl::getSequencePath(int frame) const
{
    return urlGetPath(updateUrlWithFrame(frame));
}

std::string GridAssetUrl::fullname() const
{
    std::ostringstream ss;
    // scheme + authority:
    if (!mScheme.empty())
        ss << mScheme << "://";
    // path:
    ss << mPath;
    // fragment:
    if (mGridName.length() || mIsSequence) {
        ss << "#" << mGridName;
    }
    // fragment sequence:
    if (mIsSequence) {
        if (frameStart() <= frameEnd())
            ss << "[" << frameStart() << "-" << frameEnd() << "]";
        else
            ss << "[]";
    }
    return ss.str();
}

std::string GridAssetUrl::updateUrlWithFrame(int frame) const
{
    std::string urlStr = url();
    if (mIsSequence) {

        frame = std::max(frame, mFrameStart);
        frame = std::min(frame, mFrameEnd);

        std::string tmp = urlStr;
        char        fileNameBuf[FILENAME_MAX];
        while (1) {
            auto pos = tmp.find_last_of('%');
            if (pos == std::string::npos)
                break;
            auto segment = tmp.substr(pos);
            sprintf(fileNameBuf, segment.c_str(), frame);
            segment.assign(fileNameBuf);
            tmp = tmp.substr(0, pos) + segment;
        }
        return tmp;
    }
    return urlStr;
}

void GridAssetUrl::assign(const char* s)
{
    std::string str(s);

    mScheme = urlGetScheme(str);

    std::string pathWithMeta = str;
    std::string urlNoMeta = str;
    if (mScheme.length())
        pathWithMeta = pathWithMeta.substr(mScheme.length() + 3);
    else {
        mScheme = "file";
        urlNoMeta = mScheme + "://" + urlNoMeta;
    }
    mGridName = "";
    mFrameStart = 0;
    mFrameEnd = -1;
    mIsSequence = false;

    mPath = pathWithMeta;
    auto fragment = urlGetFragment(urlNoMeta);
    if (fragment.length()) {
        // if not windows drive letter.
        mPath = pathWithMeta.substr(0, pathWithMeta.length() - fragment.length() - 1);
        urlNoMeta = mScheme + "://" + mPath;

        auto pos = fragment.find('[');
        mGridName = fragment.substr(0, pos);

        if (mPath.find('%') != std::string::npos) {
            mIsSequence = true;
        }

        if (pos != std::string::npos) {
            // found a frame range...
            auto range = fragment.substr(pos + 1);

            if ((pos = range.find('-', 0)) != std::string::npos) {
                mFrameStart = atoi(range.substr(0, pos).c_str());
                mFrameEnd = atoi(range.substr(pos + 1).c_str());
                mFrameStep = 1;
                if (mFrameEnd < mFrameStart) {
                    std::swap(mFrameEnd, mFrameStart);
                    mFrameStep = -1;
                }
            }
        }
    }
}
