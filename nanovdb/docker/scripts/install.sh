#!/bin/sh

set -e

if [ $# -lt 1 ]; then
    echo $0 {vc.zip sdk.zip target|target}
    exit 0
fi

if [ $# -eq 3 ]; then
    VC_ZIP=$(cd $(dirname $1) && pwd)/$(basename $1)
    SDK_ZIP=$(cd $(dirname $2) && pwd)/$(basename $2)
    DEST=$3
else
    DEST=$1
fi
ORIG=$(cd $(dirname $0) && pwd)

mkdir -p $DEST
cd $DEST
DEST=$(pwd)

if [ -n "$VC_ZIP" ]; then
    unzip $VC_ZIP
fi
mv VC vc
mv vc/Tools vc/tools
mv vc/tools/MSVC vc/tools/msvc
if [ -d kits/10 ]; then
    cd kits/10
else
    mkdir kits
    cd kits
    unzip $SDK_ZIP
    cd 10
fi
mv Lib lib
mv Include include
cd ../..
SDKVER=$(basename $(echo kits/10/include/* | awk '{print $NF}'))
$ORIG/lowercase kits/10/include/$SDKVER/um
$ORIG/lowercase kits/10/include/$SDKVER/shared
$ORIG/fixinclude kits/10/include/$SDKVER/um
$ORIG/fixinclude kits/10/include/$SDKVER/shared
for arch in x86 x64 arm arm64; do
    $ORIG/lowercase kits/10/lib/$SDKVER/um/$arch
done

SDKVER=$(basename $(echo kits/10/include/* | awk '{print $NF}'))
MSVCVER=$(basename $(echo vc/tools/msvc/* | awk '{print $1}'))
BASE_WIN=z:$(echo $DEST | sed 's,/,\\\\\\\\,g')
cat $ORIG/wrappers/msvcenv.sh | sed 's/MSVCVER=.*/MSVCVER='$MSVCVER/ | sed 's/SDKVER=.*/SDKVER='$SDKVER/ | sed 's/BASE=.*/BASE='"$BASE_WIN"/ | sed 's,BASE_UNIX=.*,BASE_UNIX='$DEST, > msvcenv.sh
for arch in x86 x64 arm arm64; do
    mkdir -p bin/$arch
    cp $ORIG/wrappers/* bin/$arch
    cat msvcenv.sh | sed 's/ARCH=.*/ARCH='$arch/ > bin/$arch/msvcenv.sh
done
rm msvcenv.sh
