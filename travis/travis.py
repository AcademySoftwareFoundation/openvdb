# Copyright (c) 2012-2017 DreamWorks Animation LLC
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
#
# Python script to download the latest Houdini production builds
#
# Author: Dan Bailey

import mechanize
import sys
import re
import exceptions

# this argument is for the major.minor version of Houdini to download (such as 15.0, 15.5, 16.0)
version = sys.argv[1]

if not re.match('[0-9][0-9]\.[0-9]$', version):
    raise IOError('Invalid Houdini Version "%s", expecting in the form "major.minor" such as "16.0"' % version)

br = mechanize.Browser()
br.set_handle_robots(False)

# login to sidefx.com as openvdb
br.open('https://www.sidefx.com/login/?next=/download/daily-builds')
br.select_form(nr=0)
br.form['username'] = 'openvdb'
br.form['password'] = 'L3_M2f2W'
br.submit()

# retrieve download id
br.open('http://www.sidefx.com/download/daily-builds/')

for link in br.links():
    if not link.url.startswith('/download/download-houdini'):
        continue
    if link.text.startswith('houdini-%s' % version) and 'linux_x86_64' in link.text:
        response = br.follow_link(text=link.text, nr=0)
        url = response.geturl()
        id = url.split('/download-houdini/')[-1]
        break

# accept eula terms
url = 'https://www.sidefx.com/download/eula/accept/?next=/download/download-houdini/%sget/' % id
br.open(url)
br.select_form(nr=0)
br.form.find_control('terms').items[1].selected=True
br.submit()

# download houdini tarball in 50MB chunks
url = 'https://www.sidefx.com/download/download-houdini/%sget/' % id
response = br.open(url)
mb = 1024*1024
chunk = 50
size = 0
file = open('hou.tar.gz', 'wb')
for bytes in iter((lambda: response.read(chunk*mb)), ''):
    size += 50
    print 'Read: %sMB' % size
    file.write(bytes)
file.close()
