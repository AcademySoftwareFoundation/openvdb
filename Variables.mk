openvdb_dir = $(DD_TOOLS_ROOT)/$(OS)/package/openvdb/$(PACKAGE_VERSION)
OPENVDB_PATH += -DOPENVDB_INCL_DIR=$(openvdb_dir)/include 
OPENVDB_PATH += -DOPENVDB_LIB_DIR=$(openvdb_dir)/lib 

