# Overview

This project aims to be governed in a transparent, accessible way for the benefit of the community. All participation in this project is open and not bound to corporate affiliation. Participants are all bound to the [Code of Conduct](CODE_OF_CONDUCT.md).

# Project roles

## Contributor

The contributor role is the starting role for anyone participating in the project and wishing to contribute code.

### Process for becoming a contributor

* Review the [coding standards](https://www.openvdb.org/documentation/doxygen/codingStyle.html) to ensure your contribution is in line with the project's coding and styling guidelines.
* Have a signed CLA on file ( see [below](#contributor-license-agreements) )
* Submit your code as a PR with the appropriate [DCO sign-off](#commit-sign-off).
* Have your submission approved by the [committer(s)](#committer) and merged into the codebase.

### Legal Requirements

OpenVDB is a project of the Academy Software Foundation and follows the
open source software best practice policies of the Linux Foundation.

#### License

OpenVDB is licensed under the [Mozilla Public License, version 2.0](LICENSE.md)
license. Contributions to OpenVDB should abide by that standard
license.

#### Contributor License Agreements

Developers who wish to contribute code to be considered for inclusion
in OpenVDB must first complete a **Contributor License Agreement**.

OpenVDB uses [EasyCLA](https://lfcla.com/) for managing CLAs, which automatically
checks to ensure CLAs are signed by a contributor before a commit
can be merged.

* If you are an individual writing the code on your own time and
  you're SURE you are the sole owner of any intellectual property you
  contribute, you can [sign the CLA as an individual contributor](https://docs.linuxfoundation.org/lfx/easycla/contributors/individual-contributor).

* If you are writing the code as part of your job, or if there is any
  possibility that your employers might think they own any
  intellectual property you create, then you should use the [Corporate
  Contributor Licence
  Agreement](https://docs.linuxfoundation.org/lfx/easycla/contributors/corporate-contributor).

The OpenVDB CLAs are the standard forms used by Linux Foundation
projects and [recommended by the ASWF TAC](https://github.com/AcademySoftwareFoundation/tac/blob/master/process/contributing.md#contributor-license-agreement-cla). You can review the text of the CLAs in the [TSC directory](tsc/).

#### Commit Sign-Off

Every commit must be signed off.  That is, every commit log message
must include a “`Signed-off-by`” line (generated, for example, with
“`git commit --signoff`”), indicating that the committer wrote the
code and has the right to release it under the
[Mozilla Public License, version 2.0](LICENSE.md)
license. See the [TAC documentation on contribution sign off](https://github.com/AcademySoftwareFoundation/tac/blob/master/process/contributing.md#contribution-sign-off) for more information on this requirement.

## Committer

The committer role enables the participant to commit code directly to the repository, but also comes with the obligation to be a responsible leader in the community.

### Process for becoming a committer

* Show your experience with the codebase through contributions and engagement on the community channels.
* Request to become a committer.
* Have the majority of committers approve you becoming a committer.
* Your name and email is added to the MAINTAINERS.md file for the project.

### Committer responsibilities

* Monitor email aliases.
* Monitor Slack (delayed response is perfectly acceptable).
* Triage GitHub issues and perform pull request reviews for other committers and the community.
* Make sure that ongoing PRs are moving forward at the right pace or close them.
* Remain an active contributor to the project in general and the code base in particular.

### When does a committer lose committer status?

If a committer is no longer interested or cannot perform the committer duties listed above, they
should volunteer to be moved to emeritus status. In extreme cases this can also occur by a vote of
the committers per the voting process below.

## Technical Steering Committee (TSC) member

The Technical Steering Committee (TSC) oversees the overall technical direction of OpenVDB, as defined in the [charter](charter.md).

TSC voting members consist of committers that have been nominated by the committers, with a supermajority of voting members required to have a committer elected to be a TSC voting member. TSC voting members term and succession is defined in the [charter](charter.md).

All meetings of the TSC are open to participation by any member of the OpenVDB community. Meeting times are listed in the [ASWF technical community calendar](https://lists.aswf.io/g/tac/calendar).

## Current TSC members

* Ken Museth, Chair / NVIDIA
* Andre Pradhana, DreamWorks
* Jeff Lait, SideFX
* Nick Avramoussis, WETA
* Dan Bailey, ILM
* Richard Jones, DNEG

# Release Process

Project releases will occur on a scheduled basis as agreed to by the TSC.

# Conflict resolution and voting

In general, we prefer that technical issues and committer status/TSC membership are amicably worked out
between the persons involved. If a dispute cannot be decided independently, the TSC can be
called in to decide an issue. If the TSC themselves cannot decide an issue, the issue will
be resolved by voting. The voting process is a simple majority in which each TSC receives one vote.

# Communication

This project, just like all of open source, is a global community. In addition to the [Code of Conduct](CODE_OF_CONDUCT.md), this project will:

* Keep all communication on open channels ( mailing list, forums, chat ).
* Be respectful of time and language differences between community members ( such as scheduling meetings, email/issue responsiveness, etc ).
* Ensure tools are able to be used by community members regardless of their region.

If you have concerns about communication challenges for this project, please contact the [TSC](mailto:openvdb-tsc-private@lists.aswf.io).
