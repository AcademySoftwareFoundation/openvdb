![OpenVDB](http://www.openvdb.org/images/openvdb_logo.png)


## OpenVDB Points

[Documentation](http://dneg.github.io/openvdb_points_dev)

The OpenVDB Points library extends Dreamworks' OpenVDB library to provide the ability to efficiently represent point and attribute data in VDB Grids. Points are spatially-organised into VDB voxels to provide faster access and a greater opportunity for data compression compared with linear point arrays. In using OpenVDB, this extension library can leverage the extensive toolset and active user community that has developed over the last few years.

The primary intended audience for this library is simulation and rendering for VFX studios. This is a particularly data-intensive portion of a production pipeline where the effects of improved data compression, I/O and performance are most beneficial.

An overview of this library was presented at Siggraph 2015 for which the slides can be found online at [OpenVDB Particle Storage](http://www.openvdb.org/download/openvdb_particle_storage_2015.pdf).

Please use the [OpenVDB Forum](http://www.openvdb.org/forum/) for discussion related to this project or email the developers directly at openvdbpoints@dneg.com.

[![Build Status](https://travis-ci.org/dneg/openvdb_points_dev.svg?branch=master)](https://travis-ci.org/dneg/openvdb_points_dev)
[![codecov.io](https://codecov.io/github/dneg/openvdb_points_dev/coverage.svg?branch=master)](https://codecov.io/github/dneg/openvdb_points_dev?branch=master)

### Beta Release

This library is being provided as Beta at this stage with this initial public release focusing on the core API and a relatively limited Houdini integration.

A quick summary of what is being offered:

##### OpenVDB Points:

* AttributeArray - storing and accessing of typed attribute arrays as well as handles for fast access and manipulation of those arrays.
* AttributeGroup - specialized attribute array that tracks and modifies point group membership.
* AttributeSet - sets of attribute arrays and a descriptor for storing metadata describing the contents of the set.
* IndexIterator - generic array, value and filtered iterators as well as iterator counting.
* PointDataGrid - a specialization of OpenVDB LeafNode to store and access attribute data from an AttributeSet and typedefs for PointDataTree and PointDataGrid as well as OpenVDB-compatible serialization.
* PointAttribute - tools to append, drop, rename and compress attributes in a PointDataTree.
* PointConversion - tools to convert point data into a PointDataGrid.
* PointCount - tools to count points in a PointDataTree.
* PointGroup - tools to append, drop, compact and change membership for groups in a PointDataTree.
* PointLoad - tools to explicit load delay-loaded points in a PointDataTree with optional region filtering.

##### OpenVDB Points Houdini integration:

* OpenVDB Points Convert SOP - efficiently convert back-and-forth between native Houdini points and OpenVDB Points.
* OpenVDB Points Load SOP - explicitly load delay-loaded OpenVDB Points with optional region filtering.
* SOP base class - middle-click display in Houdini for attribute data.
* GR Primitive - Houdini visualization for OpenVDB Points in the viewport.

##### OpenVDB Points Clarisse integration:

* ResourceData - create an OpenVDB Points resource and load grids.
* Geometry - new geometry object that performs VDB Points ray-tracing using a custom BVH tree.
* GeometryProperty - extract point attributes to use in Clarisse.
* Module - module to create OpenVDB Points resources and custom ray-tracing geometry or native Clarisse scatterer.

Though this library has been in use at Double Negative for some time now, it hasn't had much use outside of the studio, so we are actively looking for external adoption to help mature the toolset. Once the library has been used more widely, the intention is to integrate this directly into OpenVDB.


### Extending OpenVDB

OpenVDB Points is provided as an extension library, but the repository structure, organization of the code and coding conventions have been kept as close as possible to that of OpenVDB.

This is how the library and their dependencies are laid out:

##### OpenVDB:

* libopenvdb
* libopenvdb_houdini -> libopenvdb
* OpenVDB SOPs -> libopenvdb, libopenvdb_houdini

##### OpenVDB Points:

* libopenvdb_points -> libopenvdb
* OpenVDB Points SOPs -> libopenvdb, libopenvdb_houdini, libopenvdb_points

A PointDataGrid is a VDB Grid with a new PointDataLeaf, meaning the hierarchy remains the same as for any VDB volume but uses a different LeafNode.

Due to the native integration of OpenVDB into Houdini, lots of functionality comes for free when using OpenVDB Points. For example, on building this extra library, Houdini's file SOP and Geometry ROP can natively handle OpenVDB Points.

However, there are a few limitations to be aware of, here are two such examples:

* The middle-click menu to display OpenVDB attribute data only works when using an OpenVDB SOP or OpenVDB Points SOP as the implementation for this lives in the SOP base class (note that this is only the OpenVDB Points SOP for 0.1.0).
* OpenVDB Visualize SOP doesn't understand OpenVDB Points and thus cannot display a LeafNode visualization.


### Building and Installing OpenVDB Points

First, start out by cloning, building and installing OpenVDB:

```
git clone git://github.com/dreamworksanimation/openvdb
```

Once you are familiar with this, building and installing OpenVDB Points follows much of the same process ([Install](https://github.com/dreamworksanimation/openvdb/blob/master/openvdb/INSTALL)).

To get started, you can clone a read-only copy of our git repository:

```
git clone git://github.com/dneg/openvdb_points_dev.git
```

Or use a GitHub account to fork the OpenVDB Points git repository.


### Contributing

As with OpenVDB, we prefer code submissions in the form of pull requests to this repository, and all code should adhere to the OpenVDB [coding standards](http://www.openvdb.org/documentation/doxygen/codingStyle.html)


### Licensing

OpenVDB Points is developed and hosted by Double Negative in collaboration with Dreamworks and as such is provided under the same Mozilla Public License 2.0 as OpenVDB ([License](http://www.openvdb.org/license)).

OpenVDB Points is released under the [Mozilla Public License Version 2.0](https://www.mozilla.org/MPL/2.0/), which is a free, open source, and detailed software license developed and maintained by the Mozilla Foundation. It is a hybrid of the modified [BSD license](https://en.wikipedia.org/wiki/BSD_licenses#3-clause) and the [GNU General Public License](https://en.wikipedia.org/wiki/GNU_General_Public_License) (GPL) that seeks to balance the concerns of proprietary and open source developers.


### Acknowledgements

Double Negative:

* Dan Bailey
* James Bird
* Nick Avramoussis
* Matt Warner
* Nadine Dommanget
* Rich Jones
* Harry Biddle (ex-Dneg)

Dreamworks:

* Ken Museth
* Mihai Ald&eacute;n
* Peter Cucka
* David Hill (ex-DWA)
