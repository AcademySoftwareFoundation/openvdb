
# The parent directory of the boost/ header directory
BOOST_INCL_DIR := $(HT)/include
# The directory containing libboost_iostreams, libboost_system, etc.
BOOST_LIB_DIR := $(HDSO)

# The parent directory of the OpenEXR/ header directory
EXR_INCL_DIR := $(HT)/include
# The directory containing IlmImf
EXR_LIB_DIR := $(HDSO)

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(HT)/include
# The directory containing libtbb
TBB_LIB_DIR := $(HDSO)

# # The directory containing Python.h
# PYTHON_INCL_DIR := $(HFS)/python/include/python$(PYTHON_VERSION)
# # The directory containing libpython
# PYTHON_LIB_DIR := $(HFS)/python/lib

# # The parent directory of the blosc.h header
# # (leave blank if Blosc is unavailable)
# BLOSC_INCL_DIR ?= $(HT)/include
# # The directory containing libblosc
# BLOSC_LIB_DIR ?= $(HDSO)

ifeq ($(suffix),yes)
  OPENVDB_SUFFIX ?= h$(HOUDINI_MAJOR_RELEASE).$(HOUDINI_MINOR_RELEASE)
endif
