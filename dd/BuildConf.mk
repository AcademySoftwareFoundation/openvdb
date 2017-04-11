# Function for extracting version from the output of pk manifest --field
_get_version = $(patsubst $(strip $1)-%,%,$(filter $(strip $1)-%,$2))
#_get_version = $(filter $(strip $1)-%,$2)

# List all supported versions (parsed from the manifest)
supports_field = $(shell pk manifest --field supports)

# List all required versions (parsed from the manifest)
# Expand the expression immediately, parsing the manifest file is rather slow
requires_field := $(shell pk manifest --field requires)


# Get the versions from the manifest that are required at runtime
# Will be empty if the version isn't defined in the manifest
BLOSC_VERSION = $(call _get_version, blosc, $(requires_field))
BOOST_VERSION = $(call _get_version, boost, $(requires_field))
GLFW_VERSION = $(call _get_version, glfw, $(requires_field))
ILMBASE_VERSION = $(call _get_version, openexr, $(requires_field))
OPENEXR_VERSION = $(call _get_version, openexr, $(requires_field))
OPENVDB_VERSION = $(call _get_version, openvdb, $(requires_field))
PYTHON_VERSION = $(call _get_version, python, $(requires_field))
TBB_VERSION = $(call _get_version, tbb, $(requires_field))
ZLIB_VERSION = $(call _get_version, zlib, $(requires_field))

# Set the rest here
CMAKE_VERSION ?= 3.6.2
CPPUNIT_VERSION ?= 1.12.1
GCC_VERSION ?= 4.8.3

# Python need the short version for the include
PYTHON_MAJOR_VERSION = $(word 1,$(subst ., ,$(PYTHON_VERSION)))
PYTHON_MINOR_VERSION = $(word 2,$(subst ., ,$(PYTHON_VERSION)))
PYTHON_SHORT_VERSION = $(PYTHON_MAJOR_VERSION).$(PYTHON_MINOR_VERSION)

# Boost needs it for cmake to not pick up the system one
BOOST_MAJOR_VERSION = $(word 1,$(subst ., ,$(BOOST_VERSION)))
BOOST_MINOR_VERSION = $(word 2,$(subst ., ,$(BOOST_VERSION)))
BOOST_SHORT_VERSION = $(BOOST_MAJOR_VERSION).$(BOOST_MINOR_VERSION)

# Specify where the packages are installed
TOOLS_PACKAGE_ROOT = $(DD_TOOLS_ROOT)/$(DD_OS)/package

BLOSC_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/blosc/$(BLOSC_VERSION)
BOOST_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/boost/$(BOOST_VERSION)
CMAKE_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/cmake/$(CMAKE_VERSION)
CPPUNIT_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/cppunit/$(CPPUNIT_VERSION)
GCC_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/gcc/$(GCC_VERSION)
GLFW_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/glfw/$(GLFW_VERSION)
ILMBASE_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/ilmbase/$(ILMBASE_VERSION)
OPENEXR_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/openexr/$(OPENEXR_VERSION)
OPENVDB_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/openvdb/$(BLOSC_VERSION)
PYTHON_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/python/$(PYTHON_VERSION)
TBB_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/tbb/$(TBB_VERSION)
ZLIB_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/zlib/$(ZLIB_VERSION)
