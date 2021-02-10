**Release Process for OpenVDB**

The following assumes that the current OpenVDB library version number is 6.0.0 and the new version number is 6.1.0. Adjust for the actual version numbers as appropriate.

***Publishing the release***

- [ ] Update `CHANGES` and `doc/changes.txt` with release notes.
- [ ] Ensure `openvdb/version.h` has the correct version number.
- [ ] Ensure `doc/CMakeLists.txt` has the correct `DOXYGEN_PROJECT_NUMBER` and `vdbnamespace` alias.
- [ ] Open a pull request to merge the above changes into `openvdb/master` and verify that the CI build runs successfully.
- [ ] Draft a new [GitHub release](https://github.com/AcademySoftwareFoundation/openvdb/releases). Title it "OpenVDB 6.1.0" and tag it as `v6.1.0`.
- [ ] Go to the [GitHub docs actions](https://github.com/AcademySoftwareFoundation/openvdb/actions?query=workflow%3ADocs) for OpenVDB and manually dispatch the workflow for the `master` branch which will deploy the documentation. When the action completes, [check over the documentation](https://academysoftwarefoundation.github.io/openvdb/). Fix any errors through pull request and re-dispatch the workflow if necessary.
- [ ] Publish the GitHub draft release.

***Preparing for the next release***

- [ ] Change one or more of `OPENVDB_LIBRARY_PATCH_VERSION_NUMBER`, `OPENVDB_LIBRARY_MINOR_VERSION_NUMBER` and `OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER` in `openvdb/version.h`.  Unless it is known that the next release will include API- or ABI-breaking changes, increment only the patch number to begin with (in this case, from 6.1.0 to 6.1.1).
- [ ] In `doc/CMakeLists.txt` update `DOXYGEN_PROJECT_NUMBER` and the `@vdbnamespace` alias to match `version.h`.
- [ ] Add a "Version 6.1.1 - In development" section to `CHANGES` and to `doc/changes.txt`.  Open a pull request to merge these changes into `openvdb/master`.

***Announcing the release***

- [ ] Update the [OpenVDB website](https://github.com/AcademySoftwareFoundation/openvdb-website) with a news item announcing the release, and delete the oldest news item.  Open that page in a browser and check that the website renders correctly and that there are no broken links.
- [ ] For major and minor releases, post a release announcement to the [OpenVDB forum](https://groups.google.com/forum/#!forum/openvdb-forum). Ken typically sends out the announcement.

END
