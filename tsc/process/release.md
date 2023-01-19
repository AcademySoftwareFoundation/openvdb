# Release Process for OpenVDB

The following assumes that the current OpenVDB library version number is 6.0.0 and the new version number is 6.1.0. Adjust for the actual version numbers as appropriate. The release process is typically performed from a "release candidate" GitHub branch. This is to allow other PRs to be merged while the release process is underway.

## Performing the release

1. Create a new release candidate branch such as `v6.1.0_rc` using the branch selector on [GitHub](https://www.google.com/search?q=github+creating+branches+within+your+repository) when the release is imminent.
2. Click the "Draft a new release" button under the [GitHub releases](https://github.com/AcademySoftwareFoundation/openvdb/releases) page.
    * 2a. Create a new `v6.1.0` tag under the "Choose a tag" drop-down (Note that this tag will not be created until the release is published).
    * 2b. Select `v6.1.0_rc` under the "Target" drop-down.
    * 2c. Set "OpenVDB 6.1.0" as the title.
    * 2d. As a minimum, the body should be "See the [release notes](https://www.openvdb.org/documentation/doxygen/changes.html#v6_1_0_changes) for more details.". Include the correct link to the documentation. (Note this link will be broken until the documentation is generated in step 8). It is optional to also include a few release highlights.
    * 2e. Click the "Save draft" button. DO NOT publish the release at this stage.
3. Check out the `v6.1.0_rc` branch.
4. Ensure the root `CMakeLists.txt` has the correct version number at the top.
5. Update `CHANGES` and `doc/changes.txt` with release notes. Include any outstanding changes from the /pendingchanges directory. Run the files through a spell-check tool.
6. Update the "Version 6.1.0 - In development" section in `CHANGES` and `doc/changes.txt` to replace "In development" with the planned release date.
7. Merge these changes to `v6.1.0_rc` in a new pull request and verify that the build CI runs successfully.
8. Manually dispatch the weekly CI workflow from the `v6.1.0_rc` branch and verify that the additional checks run successfully.
9. Go to the [GitHub docs actions](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/docs.yml?query=workflow%3ADocs) for OpenVDB and manually dispatch the workflow for the `v6.1.0_rc` branch and set deploy to "docs". This will generate and add the documentation to the [OpenVDB website](https://github.com/AcademySoftwareFoundation/openvdb-website) repo and typically takes around 15 mins. When the action completes, check over the documentation on the `master` branch in the [OpenVDB website](https://github.com/AcademySoftwareFoundation/openvdb-website) repo. Fix any errors through pull request and re-dispatch the workflow if necessary.
10. Publish the GitHub draft release from the [GitHub releases](https://github.com/AcademySoftwareFoundation/openvdb/releases) page.
11. Merge the `v6.1.0_rc` branch back into `master` in a new pull request to ensure edits to release notes or any other fixes go back to master.
12. Delete the `v6.1.0_rc` branch.

## Updating master for the subsequent release

13. Fetch the latest changes in your local repository and check out the `master` branch.
14. Change one or more of `OpenVDB_MAJOR_VERSION`, `OpenVDB_MINOR_VERSION` and `OpenVDB_PATCH_VERSION` in the root `CMakeLists.txt`.  Unless it is known that the next release will include API- or ABI-breaking changes, increment only the patch number to begin with (in this case, from 6.1.0 to 6.1.1).
15. Add a "Version 6.1.1 - In development" section to `CHANGES` and to `doc/changes.txt`.
16. Merge these changes to `master` in a new pull request.

## Announcing the release

17. Update the [OpenVDB website](https://github.com/AcademySoftwareFoundation/openvdb-website) with a news item announcing the release, and delete the oldest news item.  Open that page in a browser and check that the website renders correctly and that there are no broken links.
18. For major and minor releases, post a release announcement to the [OpenVDB forum](https://groups.google.com/forum/#!forum/openvdb-forum). Ken typically sends out the announcement.

END
