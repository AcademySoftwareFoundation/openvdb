.. _openvdb_houdini_add_version:

========================================
Add Houdini version to installed package
========================================

Houdini tend to break their abi quite often and instead of installing
a new version every time this happens you can choose to patch the
current installed version. To do this simply run this in
openvdb/dd/openvdb_houdini

.. code-block:: sh
                
   make add-houdini-version-<version>

Where <version> is the houdini version you want to add. It will not
automatically install the extension but will prompt you with the
command to do so when it is done.
