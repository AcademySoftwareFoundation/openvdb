#!/usr/bin/env bash

set -ex

HOUDINI_MAJOR="$1"

# install houdini pre-requisites
apt-get install -y libxi-dev
apt-get install -y csh
apt-get install -y default-jre
apt-get install -y python-mechanize

export PYTHONPATH=${PYTHONPATH}:/usr/lib/python2.7/dist-packages
# download and unpack latest houdini headers and libraries from daily-builds
python ci/download_houdini.py $HOUDINI_MAJOR

tar -xzf hou.tar.gz
ln -s houdini* hou
cd hou
tar -xzf houdini.tar.gz

cd -
