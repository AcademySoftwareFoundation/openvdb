#!/bin/bash

SDK=kits\\10
SDK_UNIX=kits/10
BASE=z:\\opt\\msvc
BASE_UNIX=/opt/msvc
MSVCVER=14.13.26128
SDKVER=10.0.16299.0
ARCH=x86
MSVCDIR="$BASE\\vc\\tools\\msvc\\$MSVCVER"
SDKINCLUDE="$BASE\\$SDK\\include\\$SDKVER"
SDKLIB="$BASE\\$SDK\\lib\\$SDKVER"
BINDIR=$BASE_UNIX/vc/tools/msvc/$MSVCVER/bin/Hostx64/$ARCH
SDKBINDIR=$BASE_UNIX/$SDK_UNIX/bin/$SDKVER/x64
export INCLUDE="$MSVCDIR\\include;$SDKINCLUDE\\shared;$SDKINCLUDE\\ucrt;$SDKINCLUDE\\um;$SDKINCLUDE\\winrt"
export LIB="$MSVCDIR\\lib\\$ARCH;$SDKLIB\\ucrt\\$ARCH;$SDKLIB\\um\\$ARCH"
export LIBPATH="$LIB"
# "$MSVCDIR\\bin\\Hostx64\\x64" is included in PATH for DLLs.
export WINEPATH="${BINDIR//\//\\};${SDKBINDIR//\//\\};$MSVCDIR\\bin\\Hostx64\\x64"
export WINEDLLOVERRIDES="vcruntime140_1=n"
