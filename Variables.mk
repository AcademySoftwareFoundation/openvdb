require_field = $(shell pk manifest --field requires)
OPENVDB_VERSION = $(patsubst openvdb-%,%,$(filter openvdb-%,$(require_field)))

openvdb_dir = $(DD_TOOLS_ROOT)/$(OS)/package/openvdb/$(OPENVDB_VERSION)
OPENVDB_PATH += -DOPENVDB_INCL_DIR=$(openvdb_dir)/include 
OPENVDB_PATH += -DOPENVDB_LIB_DIR=$(openvdb_dir)/lib 

