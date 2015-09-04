                               OpenVDB Points
========================================================================

The OpenVDB Points library extends Dreamworks' OpenVDB library to provide the
ability to efficiently represent point and attribute data in VDB Grids. Points
are spatially-organised into VDB voxels to provide faster access and a greater
opportunity for data compression compared with linear point arrays. By building
on top of OpenVDB, this library can re-use a lot of the work already in place
for the large, open-source OpenVDB toolset and active community of users.

The primary intended audience for this library is simulation and rendering for
VFX studios. This is a particularly data-intensive portion of a production
pipeline where the effects of improved data compression, I/O and performance
are most beneficial.

An overview of this library was presented at Siggraph 2015 for which the slides
can be found online at
www.openvdb.org/download/openvdb_particle_storage_2015.pdf

                               Beta Release
========================================================================

This library is being provided as Beta at this stage with this initial public
release focusing on the core API and a relatively limited Houdini integration.

A quick summary of what is being offered:

OpenVDB Points:

* AttributeArray and AttributeSet - storing and accessing of typed attribute
arrays and sets of these arrays.
* PointDataGrid - a specialization of OpenVDB LeafNode to store and access
attribute data from an AttributeSet and typedefs for PointDataTree and
PointDataGrid as well as OpenVDB-compatible serialization.
* PointConversion - a tool to convert generic point data into a PointDataGrid.

OpenVDB Points Houdini-integration:

* OpenVDB Points SOP - efficiently convert back-and-forth between native
Houdini points and OpenVDB Points.
* SOP base class - middle-click display in Houdini for attribute data.
* GR Primitive - Houdini visualization for OpenVDB Points in the viewport.

Though this library has been in use at Double Negative for some time now, it
hasn't had much use outside of the studio, so we are actively looking for
external adoption to help mature the toolset.
Once the library has been used more widely, the intention is to integrate this
directly into OpenVDB.

                               Extending OpenVDB
========================================================================

OpenVDB Points is provided as an extension library, but the repository
structure, organization of the code and coding conventions have been kept as
close as possible to that of OpenVDB.

This is how the library and their dependencies are laid out:

OpenVDB:

* libopenvdb
* libopenvdb_houdini -> libopenvdb
* OpenVDB SOPs -> libopenvdb, libopenvdb_houdini

OpenVDB Points:

* libopenvdb_points -> libopenvdb
* libopenvdb_points_houdini -> libopenvdb, libopenvdb_points
* OpenVDB Points SOPs -> libopenvdb, libopenvdb_houdini, libopenvdb_points,
                         libopenvdb_points_houdini

A PointDataGrid is a VDB Grid with a new PointDataLeaf, meaning the hierarchy
remains the same as for any VDB volume but uses a different LeafNode.

Due to the native integration of OpenVDB into Houdini, lots of functionality
comes for free when using OpenVDB Points. For example, on building this
extra library, Houdini's file SOP and Geometry ROP can natively handle OpenVDB
Points.

There are a few limitations to be aware of, here are two such examples:

* The middle-click menu to display OpenVDB attribute data only works when using
an OpenVDB Points SOP as the implementation for this lives in the SOP base
class.
* OpenVDB Visualize SOP doesn't understand OpenVDB Points and thus cannot
display a LeafNode visualization.

                               Building and Installing OpenVDB Points
========================================================================

First, start out by cloning, building and installing OpenVDB:

git clone git://github.com/dreamworksanimation/openvdb
https://github.com/dreamworksanimation/openvdb/blob/master/openvdb/INSTALL

Once you are familiar with this, building and installing OpenVDB Points
follows much of the same process.

To get started, you can clone a read-only copy of our git repository:

git clone git://github.com/dneg/openvdb_points_dev.git

Or use a GitHub account to fork the OpenVDB Points git repository.

                               Contributing
========================================================================

The main difference in development process between the two projects is in the
use of source control. While OpenVDB is primarily developed using Accurev and
periodically synced to Git, OpenVDB Points is primarily developed using Git
with a full Git history. However, both projects encourage external
contributions.

Feel free to fork the GitHub repo and create pull requests.

                               Licensing
========================================================================

OpenVDB Points is developed and hosted by Double Negative in collaboration with
Dreamworks and as such is provided under the same Mozilla Public License 2.0 as
OpenVDB:

http://www.openvdb.org/license

                               Acknowledgements
========================================================================

Dan Bailey (Dneg)
Mihai Ald&eacute;n (DWA)
Nick Avramoussis (Dneg)
Matt Warner (Dneg)
Harry Biddle (Dneg)
Peter Cucka (DWA)
Ken Museth (DWA)
