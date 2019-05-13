![OpenVDB](http://www.openvdb.org/images/openvdb_logo.png)

[![CircleCI](https://circleci.com/gh/AcademySoftwareFoundation/openvdb.svg?style=svg)](https://circleci.com/gh/AcademySoftwareFoundation/openvdb)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2774/badge)](https://bestpractices.coreinfrastructure.org/projects/2774)

[Website](http://www.openvdb.org) |
[Discussion Forum](http://www.openvdb.org/forum) |
[Documentation](http://www.openvdb.org/documentation/)

OpenVDB is an open source C++ library comprising a novel hierarchical data structure and a large suite of tools for the efficient storage and manipulation of sparse volumetric data discretized on three-dimensional grids. It was developed by DreamWorks Animation for use in volumetric applications typically encountered in feature film production.


### Development Repository

This GitHub repository hosts the trunk of the OpenVDB development. This implies that it is the newest public version with the latest features and bug fixes. However, it also means that it has not undergone a lot of testing and is generally less stable than the [production releases](https://github.com/AcademySoftwareFoundation/openvdb/releases).


### License

OpenVDB is released under the [Mozilla Public License Version 2.0](https://www.mozilla.org/MPL/2.0/), which is a free, open source, and detailed software license developed and maintained by the Mozilla Foundation. It is a hybrid of the modified [BSD license](https://en.wikipedia.org/wiki/BSD_licenses#3-clause) and the [GNU General Public License](https://en.wikipedia.org/wiki/GNU_General_Public_License) (GPL) that seeks to balance the concerns of proprietary and open source developers.


### Contributing

Developers who wish to contribute code to be considered for inclusion in the OpenVDB distribution must first complete the [Contributor License Agreement](http://www.openvdb.org/download/OpenVDBContributorLicenseAgreement.pdf) and submit it to [openvdb@gmail.com](mailto:openvdb@gmail.com).  We prefer code submissions in the form of pull requests to this repository.  All code should adhere to the OpenVDB [coding standards](http://www.openvdb.org/documentation/doxygen/codingStyle.html), and _every commit must be signed off_.  That is, every commit log message must include a “`Signed-off-by`” line (generated, for example, with “`git commit --signoff`”), indicating that the committer wrote the code and has the right to release it under the [MPL 2.0](https://www.mozilla.org/MPL/2.0/) license. See http://developercertificate.org/ for more information on this requirement.


### Developer Quick Start

#### Linux
##### Installing Dependencies (Boost, TBB, OpenEXR, Blosc)

```
apt-get install -y libboost-iostreams-dev
apt-get install -y libboost-system-dev
apt-get install -y libtbb-dev
apt-get install -y libilmbase-dev
apt-get install -y libopenexr-dev
```
```
git clone git@github.com:Blosc/c-blosc.git
cd c-blosc
git checkout tags/v1.5.0 -b v1.5.0
mkdir build
cd build
cmake ..
make -j4
make install
cd ../..
```

##### Building OpenVDB
```
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake ..
make -j4
make install
```
#### Mac OSX
##### Installing Dependencies (Boost, TBB, OpenEXR, Blosc)
```
brew install boost
brew install tbb
brew install ilmbase
brew install openexr
```
```
git clone git@github.com:Blosc/c-blosc.git
cd c-blosc
git checkout tags/v1.5.0 -b v1.5.0
mkdir build
cd build
cmake ..
make -j4
make install
cd ../..
```
##### Building OpenVDB
```
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake ..
make -j4
make install
```
#### Windows
##### Installing Dependencies (Boost, TBB, OpenEXR, Blosc)
```
vcpkg install boost zlib blosc openexr tbb
vcpkg integrate install
```
##### Building OpenVDB
```
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_VCPKG>\scripts\buildsystems\vcpkg.cmake ..
make -j4
make install
```
