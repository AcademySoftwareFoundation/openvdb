#!/usr/bin/env bash

set -ex

MSVC_VERSION="$1"; shift

dos2unix ./scripts/vsdownload.py ./scripts/install.sh ./scripts/lowercase ./scripts/fixinclude
chmod +x ./scripts/vsdownload.py ./scripts/install.sh ./scripts/lowercase ./scripts/fixinclude

./scripts/vsdownload.py --accept-license --dest /opt/msvc --msvc-${MSVC_VERSION}

./scripts/install.sh /opt/msvc

find /opt/msvc/bin -type f -exec chmod +x {} \;
find /opt/msvc/bin -type f -exec dos2unix {} \;
