# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Simplest possible Makefile, which provides all the services in the included
# Makefile.tdpackage. To see all the options provided, just type 'make'.
# It is recommend to add comments for new things you would be adding
# to keep this file readable for other people who may need to maintain
# your package in the future.

# The version number of the facility's Makefile this package is making use of.
USE_MAKEFILE_VERSION = 4

# Openvdb's docs are online
NO_DOCS=true

# No support for git so disabling it
NO_TAG=true

# This Makefile defines various build targets and variables, taking into
# account your settings, above. Do not remove this line unless you really
# know what you are doing.
include $(DD_TOOLS_ROOT)/etc/Makefile.tdpackage

