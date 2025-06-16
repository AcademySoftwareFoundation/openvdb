// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/Viewport.h

    \author Petra Hapalova

    \brief
*/

#include <imgui/ImguiWindow.h>

namespace pnanovdb_editor
{
class Viewport
{
public:
    // singleton
    static Viewport& getInstance()
    {
        static Viewport instance;
        return instance;
    }

    Viewport();
    void setup();
    void render(const char* title);

private:

};
}
