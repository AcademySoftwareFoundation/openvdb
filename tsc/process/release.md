**Release Process for OpenVDB**

The following assumes that the current OpenVDB library version number is 6.0.0 and the new version number is 6.1.0. Adjust for the actual version numbers as appropriate. The release process is typically performed from a "release candidate" GitHub branch, but can also be done from a master branch.

***Creating the release candidate branch***

- [ ] Create a new release candidate branch such as `v6.1.0_rc` when the release is imminent.
- [ ] Under the `master` branch, change one or more of `OpenVDB_MAJOR_VERSION`, `OpenVDB_MINOR_VERSION` and `OpenVDB_PATCH_VERSION` in the root `CMakeLists.txt`.  Unless it is known that the next release will include API- or ABI-breaking changes, increment only the patch number to begin with (in this case, from 6.1.0 to 6.1.1).
- [ ] Add a "Version 6.1.1 - In development" section to `CHANGES` and to `doc/changes.txt`.  Open a pull request to merge these changes into `master`.

***Publishing the release***

- [ ] Check out the release candidate branch.
- [ ] Update `CHANGES` and `doc/changes.txt` with release notes. Include any outstanding from the /pendingchanges directory.
- [ ] Ensure the root `CMakeLists.txt` has the correct version number at the top.
- [ ] Open a pull request to merge the above changes into the release candidate branch and verify that the CI build runs successfully.
- [ ] Draft a new [GitHub release](https://github.com/AcademySoftwareFoundation/openvdb/releases) from the release candidate branch. Title it "OpenVDB 6.1.0" and tag it as `v6.1.0`.
- [ ] Go to the [GitHub docs actions](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/docs.yml?query=workflow%3ADocs) for OpenVDB and manually dispatch the workflow for the release candidate branch which will deploy the documentation. When the action completes, [check over the documentation](https://academysoftwarefoundation.github.io/openvdb/). Fix any errors through pull request and re-dispatch the workflow if necessary.
- [ ] Publish the GitHub draft release.
- [ ] Merge the release candidate branch back into master in a new pull request to ensure edits to release notes or any other fixes go back to master.

***Announcing the release***

- [ ] Update the [OpenVDB website](https://github.com/AcademySoftwareFoundation/openvdb-website) with a news item announcing the release, and delete the oldest news item.  Open that page in a browser and check that the website renders correctly and that there are no broken links.
- [ ] For major and minor releases, post a release announcement to the [OpenVDB forum](https://groups.google.com/forum/#!forum/openvdb-forum). Ken typically sends out the announcement.

END
