MAYA_HOME := $(PACKAGES_DIR)/maya/$(MAYA_VERSION)
MAYA_INC_DIR := $(MAYA_HOME)/include
MAYA_LIB_DIR := $(MAYA_HOME)/lib

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(MAYA_INC_DIR)
# The directory containing libtbb
TBB_LIB_DIR := $(MAYA_LIB_DIR)

ifeq ($(suffix),yes)
  OPENVDB_SUFFIX ?= m$(MAYA_VERSION)
endif
