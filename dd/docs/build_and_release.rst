.. _openvdb_build_and_release:

=================
Build and release
=================

Building and releasing opendvb is fairly straight forward, but there
are some caveats that the builder needs to be aware of. First of
openvdb is split up into four packages:

- openvdb: The core package, contains all functionality that defines OpenVDB.
- openvdb_houdini: Houdini plugins for OpenVDB as wells as houdini
  specific libraries.
- openvdb_maya: Maya plugins for OpenVDB.
- pyopenvdb: Python bindings for OpenVDB.

Reason behind spliting openvdb into all these packages is to make it
easier to maintain. For example if a new python version is needed,
only the pyopenvdb package needs to be recomplied and released. Same
goes if a bug is found in openvdb, only that package needs to be
recompiled and released.

This however adds some complexity for building and releasing
them. Which this documentation will try and sort out.

Preparing build environment
---------------------------

To build any of the four openvdb packages, you need to get the
source. At the moment it is on github and is a fork of Dreamworks
Animation's openvdb repo.

To get it simply clone the repo:

.. code-block:: sh

   git clone https://github.com/digitaldomain/openvdb.git

Build
-----

Compared to the original repository, the fork adds a dd directory
containing all the files for building and installing the packages on
our system.

There is one restriction on the build order for openvdb. The core
package needs to be installed before you can build any of the other
three. This because all three depends on the core library.

To build one of the packages, go to the directory matching the package
name in the dd directory and type "make build". For example if you
want to build the core package.

.. code-block:: sh
                
   cd openvdb/dd/openvdb
   make build

Release
-------

Releasing the packages are quite similar to building them. Go the
directory matching the package name in the dd directory and type "make
install".

.. code-block:: sh

   # Install it to user workspace
   cd openvdb/dd/<package name>
   make install

If you want to install it to facility append "CONTEXT=facility".

.. code-block:: sh
                
   # Install it to facility
   cd openvdb/dd/<package name>
   make install CONTEXT=facility

.. note:: You do not need to run "make build" before "make install". 


Changing versions
-----------------

OpenVDB depends on a few other third party packages and updating the
versions for those are handled in two locations. Packages required at
runtime are set in the manifest file (manifest.yaml) under the
Requires field for the specific openvdb package. These are local to
the openvdb package. For example changing a version in the manifest
for openvdb_maya will not affect openvdb_houdini and vice versa. 

.. note:: You cannot use ranges or anything other fancy in the
          requires field. Since the makefile simply just parse the
          values in the field.

          For example:

          .. code-block:: yaml
                
             Requires:
             - tbb-4.4+
               ...

          Will set the tbb version to 4.4+ and cmake will fail with an
          error about not finding tbb.

Then there are third party packages that are only needed at build
time, these are cmake, gcc and cppunit. They are updated either in the
BuildConf.mk file, which is located at the root of the dd
directory. Or passed in as commandline arguments when building. For
example use cmake 3.7.2 when building you can do:

.. code-block:: sh
                
   make build CMAKE_VERSION=3.7.2

These are global settings, which means if updated in the BuildConf.mk
file it will affect all packages of openvdb.

Specify versions of maya/houdini to build for
---------------------------------------------

In the packages openvdb_houdini and openvdb_maya you list what
versions of respectively program you want to build for in the Supports
field in the manifest.

For example build openvdb_houdini for houdini 16.0.565 and 16.0.535:

.. code-block:: yaml
                
   # openvdb_houdini's manifest.yaml
   Supports:
   - houdini-16.0.535
   - houdini-16.0.565

And build openvdb_maya for 2015.5 and 2016:

.. code-block:: yaml
                
   # openvdb_maya's manifest.yaml
   Supports:
   - maya-2015.5
   - maya-2016

.. note:: For openvdb_maya you only build for the major releases. To
          get what major version a specific maya version has, you can
          use

          .. code-block:: sh
                          
             pk-maya-version --major <version>

          Where <version> is the maja version.


Gotchas
-------

There are a few gotchas you need to look out for when building all the
openvdb packages. You cannot simply build one openvdb core package and
link that with the rest. That would be too easy.

Here follows what you need to think about for the different packages. 

Houdini
~~~~~~~

Houdini ships with it's own plugins for openvdb. Therefore you need to
make sure that the user can pass data between those plugins and the
one you are building.

Both 15 and 16 are still on openvdb 3. Which means that if you just
build openvdb 4.0 with it's default settings, things will
break. Instead you need to use the abi 3 compatibility option when
building. This will be automatically enable when you use the _abi3
flavor in the version name for the core package of openvdb.


Then there are the third party libraries used to build the core
package. They must match the ones houdini ships, otherwise prepare for
angry mob with pitchforks and torches!

The third party libraries that must match are tbb, boost, blosc and
ilmbase/openexr. The versions for boost, blosc and ilmbase have been
pretty stable for a while. Houdini 15 and 16 both uses

- blosc 1.5.0
- boost 1.55.0
- ilmbase/openexr 2.2.0

Even though `VFX Reference Platform <http://www.vfxplatform.com>`_
specifies boost 1.61 for 2017.

However Houdini 16 moved to tbb 4.4 where as Houdini 15 uses 4.3. Keep
this in mind when building. 

Here is an example of what you would need to build for Houdini 15.5
and Houdini 16 when you are building version 4.0.1.

openvdb-4.0.1_abi3_tbb43 

- Has abi 3 compatibility turned on
- Uses tbb 4.3.x
- Rest follows the `VFX Reference Platform <http://www.vfxplatform.com>`_

openvdb-4.0.1_abi3

- Has abi 3 compatibility turned on
- Uses tbb 4.4.x but since this is the standard for 2017 on `VFX
  Reference Platform <http://www.vfxplatform.com>`_, you do not need
  to add a flavor tag for it.

openvdb_houdini-4.0.1_h15

- Depends on openvdb-4.0.1_abi3_tbb43 
- Only built for Houdini 15 versions
- Added flavor tag since Houdini 15 is an older houdini version.

openvdb_houdini-4.0.1

- Depends on openvdb-4.0.1_abi3
- Only built for Houdini 16 versions.
- No flavor tag needed since this is the main version.

Maya
~~~~

When building the plugins for maya you only need to look out for tbb,
which is shipped with maya. Autodesk are pretty good at following the
`VFX Reference Platform <http://www.vfxplatform.com>`_. 
