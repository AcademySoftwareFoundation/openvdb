#!/usr/bin/env bash

set -ex

HOUDINI_MAJOR="$1"
PLATFORM="$2"
OTHER_ARGS="$3"

pip install --user requests
python ci/download_houdini.py $HOUDINI_MAJOR $PLATFORM $OTHER_ARGS

if [[ $PLATFORM =~ "linux" ]]; then
    # create dir hierarchy
    mkdir -p hou/bin
    mkdir -p hou/houdini
    mkdir -p hou/toolkit
    mkdir -p hou/dsolib

    # unpack hou.tar.gz and cleanup
    tar -xzf hou.tar.gz
    rm -rf hou.tar.gz
    cd houdini*
    tar -xzf houdini.tar.gz

    # copy required files into hou dir
    cp houdini_setup* ../hou/.

    # report library names
    ls -al dsolib/

    # copy required libraries
    cp -r toolkit/cmake ../hou/toolkit/.
    cp -r toolkit/include ../hou/toolkit/.
    cp -r dsolib/libHoudini* ../hou/dsolib/.
    cp -r dsolib/libopenvdb_sesi* ../hou/dsolib/.
    cp -r dsolib/libblosc* ../hou/dsolib/.
    cp -r dsolib/libhboost* ../hou/dsolib/.
    cp -r dsolib/libz* ../hou/dsolib/.
    cp -r dsolib/libbz2* ../hou/dsolib/.
    cp -r dsolib/libtbb* ../hou/dsolib/.
    cp -r dsolib/libjemalloc* ../hou/dsolib/.
    cp -r dsolib/liblzma* ../hou/dsolib/.
    cp -r dsolib/libIex* ../hou/dsolib/.
    cp -r dsolib/libImath* ../hou/dsolib/.
    cp -r dsolib/libIlmThread* ../hou/dsolib/.
    cd ..

elif [[ $PLATFORM =~ "macos" ]]; then
    # Exract files by mounting the downloaded dmg (we only really want to
    # expand Houdini.framework)
    hdiutil attach hou.dmg
    pkgutil --expand-full /Volumes/Houdini/Houdini.pkg Houdini
    hdiutil detach /Volumes/Houdini
    rm hou.dmg

    # Move the required Frameworks and delete the extracted src
    mkdir -p hou/Frameworks
    mv Houdini/Framework.pkg/Payload/Houdini.framework hou/Frameworks/Houdini.framework
    rm -rf Houdini

    # Report library names
    ls -al hou/Frameworks/Houdini.framework/Libraries

    # Remove unused resources
    cd hou/Frameworks/Houdini.framework/Resources/
    rm -rf $(ls | grep -e toolkit -v)
    cd -

    # Handle libraries. On some versions of MacOS with older versions of ld,
    # ld will complain (error) if shared libraries contain missing files which
    # are referenced with LC_LOAD_DYLIB or LC_RPATH entries (even though they
    # are not explicitly required at link time). We still want to delete these
    # unused libs as they occupt ~1-2GB. To handle this, we generate a unique
    # list of libs that our direct dependencies reference and create an empty
    # shared dylib in their place.
    cd hou/Frameworks/Houdini.framework/Libraries
    # Remove any folders here, they aren't needed
    rm -rf $(ls -p | grep /)
    # Remove any library that does not match the -e patterns (inverse grep with -v)
    unused_libraries=$(ls | \
        grep -e libHoudini \
             -e libopenvdb_sesi \
             -e libblosc \
             -e libhboost \
             -e libz \
             -e libbz2 \
             -e libtbb \
             -e libjemalloc \
             -e liblzma \
             -e libIex \
             -e libImath \
             -e libIlmThread \
             -v)
    rm -rf ${unused_libraries}

    # Create an empty valid shared lib
    echo '' | clang -x c -shared -o libempty.dylib -

    # Generate a unique list of libs that our remaining libs reference
    for i in $(ls); do otool -LX $i >> libnames; done
    sort -u libnames | grep @rpath | cut -f1 -d' ' | xargs > rpaths

    # Recreate unused libraries that have been deleted as empty shared dylibs
    # to keep ld happy
    for libpath in $(cat rpaths); do
        libpath=${libpath#"@rpath/"}
        echo "Checking $libpath"
        if [ ! -f $libpath ]; then
            echo "Creating empty library at $libpath"
            mkdir -p $(dirname $libpath)
            cp libempty.dylib $libpath
        fi
    done

    rm libempty.dylib
    cd -
fi

# write hou into hou.tar.gz and cleanup
tar -czvf hou.tar.gz hou

# move hou.tar.gz into hou subdirectory
rm -rf hou/*
mv hou.tar.gz hou

# inspect size of tarball
ls -lart hou/hou.tar.gz
