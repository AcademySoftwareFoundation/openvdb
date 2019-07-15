# Copyright (c) DreamWorks Animation LLC
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DreamWorks Animation nor the names of
# its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
# LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.

# Startup script to set the visibility of (and otherwise customize)
# open-source (ASWF) OpenVDB nodes and their native Houdini equivalents
#
# To be installed as <dir>/python2.7libs/pythonrc.py,
# where <dir> is a path in $HOUDINI_PATH.

import hou
import os


# Construct a mapping from ASWF SOP names to names of equivalent
# native Houdini SOPs.
sopcategory = hou.sopNodeTypeCategory()
namemap = {}
for name, sop in sopcategory.nodeTypes().items():
    try:
        nativename = sop.spareData('nativename')
        if nativename:
            namemap[name] = nativename
    except AttributeError:
        pass

# Print the list of correspondences.
#from pprint import pprint
#pprint(namemap)


# Determine which VDB SOPs should be visible in the Tab menu:
# - If $OPENVDB_OPHIDE_POLICY is set to 'aswf', hide AWSF SOPs for which
#   a native Houdini equivalent exists.
# - If $OPENVDB_OPHIDE_POLICY is set to 'native', hide native Houdini SOPs
#   for which an ASWF equivalent exists.
# - Otherwise, show both the ASWF and the native SOPs.
names = []
ophide = os.getenv('OPENVDB_OPHIDE_POLICY', 'none').strip().lower()
if ophide == 'aswf':
    names = namemap.keys()
elif ophide == 'native':
    names = namemap.values()

for name in names:
    sop = sopcategory.nodeType(name)
    if sop:
        sop.setHidden(True)


# Customize SOP visibility with code like the following:
#
#     # Hide the ASWF Clip SOP.
#     sopcategory.nodeType('DW_OpenVDBClip').setHidden(True)
#
#     # Show the native VDB Clip SOP.
#     sopcategory.nodeType('vdbclip').setHidden(False)
#
#     # Hide all ASWF advection SOPs for which a native equivalent exists.
#     for name in namemap.keys():
#         if 'Advect' in name:
#             sopcategory.nodeType(name).setHidden(True)

# Copyright (c) DreamWorks Animation LLC
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
