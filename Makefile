.PHONY: all
all: 
	@cd openvdb && $(MAKE) -f Package.mk

# Call the real makefiles
.PHONY: %
%: 
	@cd openvdb && $(MAKE) -f Package.mk $@ && cd - && \
	cd openvdb_houdini && $(MAKE) -f Package.mk $@ && cd - && \
	cd openvdb_maya && $(MAKE) -f Package.mk $@ && cd -

.PHONY: help
help: 
	@cd openvdb && $(MAKE) -f Package.mk $@
